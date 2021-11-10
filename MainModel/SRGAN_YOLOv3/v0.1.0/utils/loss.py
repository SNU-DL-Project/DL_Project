import numpy as np
import math

import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16

from .utils import to_cpu
from .yoloutils import bbox_iou, bbox_iou_mod


def compute_yolo_loss(predictions, targets, model):
    """
    1. compute_loss
    loss를 계산하는 함수

    predictions : 추정값
    targets : True 값
    model : loss를 계산할 모델

    loss를 return
    """
    # device check
    device = targets.device

    # 3가지 loss가 존재 : class, box, object
    loss_cls = torch.zeros(1, device=device)
    loss_box = torch.zeros(1, device=device)
    loss_obj = torch.zeros(1, device=device)

    # yolo target을 build
    tcls, tbox, indices, anchors = build_targets(predictions, targets, model)  # targets

    # class, object에 대한 loss function을 정의
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))

    # 각 yolo layer에 대해 loss를 계산
    for layer_index, layer_predictions in enumerate(predictions):
        # Get image ids, anchors, grid index i and j for each target in the current yolo layer
        b, anchor, grid_j, grid_i = indices[layer_index]

        # Build empty object target tensor with the same shape as the object prediction
        tobj = torch.zeros_like(layer_predictions[..., 0], device=device)  # target obj

        # Get the number of targets for this layer.
        # Each target is a label box with some scaling and the association of an anchor box.
        # Label boxes may be associated to 0 or multiple anchors. So they are multiple times or not at all in the targets.
        num_targets = b.shape[0]

        # Check if there are targets for this batch
        if num_targets:
            # Load the corresponding values from the predictions for each of the targets
            ps = layer_predictions[b, anchor, grid_j, grid_i]

            # Regression of the box
            # Apply sigmoid to xy offset predictions in each cell that has a target
            pxy = ps[:, :2].sigmoid()
            # Apply exponent to wh predictions and multiply with the anchor box that matched best with the label for each cell that has a target
            pwh = torch.exp(ps[:, 2:4]) * anchors[layer_index]
            # Build box out of xy and wh
            pbox = torch.cat((pxy, pwh), 1)
            # Calculate CIoU or GIoU for each target with the predicted box for its cell + anchor
            #iou = bbox_iou(pbox.T, tbox[layer_index], x1y1x2y2=False, CIoU=True)
            iou = bbox_iou_mod(pbox.T, tbox[layer_index], x1y1x2y2=False)

            # We want to minimize our loss so we and the best possible IoU is 1 so we take 1 - IoU and reduce it with a mean
            loss_box += (1.0 - iou).mean()  # iou loss

            # Classification of the objectness
            # Fill our empty object target tensor with the IoU we just calculated for each target at the targets position
            tobj[b, anchor, grid_j, grid_i] = iou.detach().clamp(0).type(tobj.dtype)  # Use cells with iou > 0 as object targets

            # Classification of the class
            # Check if we need to do a classification (number of classes > 1)
            if ps.size(1) - 5 > 1:
                # Hot one class encoding
                t = torch.zeros_like(ps[:, 5:], device=device)  # targets
                t[range(num_targets), tcls[layer_index]] = 1
                # Use the tensor to calculate the BCE loss
                loss_cls += BCEcls(ps[:, 5:], t)  # BCE

        # Classification of the objectness the sequel
        # Calculate the BCE loss between the on the fly generated target and the network prediction
        loss_obj += BCEobj(layer_predictions[..., 4], tobj) # obj loss

    loss_box *= 0.05
    loss_obj *= 1.0
    loss_cls *= 0.5

    # Merge losses
    loss = loss_box + loss_obj + loss_cls

    return loss, to_cpu(torch.cat((loss_box, loss_obj, loss_cls, loss)))

def build_targets(p, targets, model):
    """
    2. build_targets
    target variable


    """
    # input target(image,class,x,y,w,h)에 대한
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    # anchors_num : anchor 수, yolov3 기반 모델은 모두 3이니, 신경쓰지 않아도 된다.
    # targets_num : target의 개수
    anchors_num, targets_num = 3, targets.shape[0]  # number of anchors, targets

    # tcls : target class
    # tbox : target box
    # indices :
    # anchors_list : anchor들의 list
    tcls, tbox, indices, anchors_list = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain

    # Make a tensor that iterates 0-2 for 3 anchors and repeat that as many times as we have target boxes
    ai = torch.arange(anchors_num, device=targets.device).float().view(anchors_num, 1).repeat(1, targets_num)
    # Copy target boxes anchor size times and append an anchor index to each copy the anchor index is also expressed by the new first dimension
    targets = torch.cat((targets.repeat(anchors_num, 1, 1), ai[:, :, None]), 2)

    for i, yolo_layer in enumerate(model.yolo_layers):
        # Scale anchors by the yolo grid cell size so that an anchor with the size of the cell would result in 1
        anchors = yolo_layer.anchors / yolo_layer.stride

        # Add the number of yolo cells in this layer the gain tensor
        # The gain tensor matches the collums of our targets (img id, class, x, y, w, h, anchor id)

        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Scale targets by the number of yolo layer cells, they are now in the yolo cell coordinate system
        t = targets * gain

        # Check if we have targets
        if targets_num:
            # Calculate ration between anchor and target box for both width and height
            r = t[:, :, 4:6] / anchors[:, None]
            # Select the ratios that have the highest divergence in any axis and check if the ratio is less than 4
            j = torch.max(r, 1. / r).max(2)[0] < 4  # compare #TODO
            # Only use targets that have the correct ratios for their anchors
            # That means we only keep ones that have a matching anchor and we loose the anchor dimension
            # The anchor id is still saved in the 7th value of each target
            t = t[j]
        else:
            t = targets[0]

        # Extract image id in batch and class id
        b, c = t[:, :2].long().T

        # We isolate the target cell associations.
        # x, y, w, h are allready in the cell coordinate system meaning an x = 1.2 would be 1.2 times cellwidth
        gxy = t[:, 2:4]
        gwh = t[:, 4:6]  # grid wh

        # Cast to int to get an cell index e.g. 1.2 gets associated to cell 1
        gij = gxy.long()

        # Isolate x and y index dimensions
        gi, gj = gij.T  # grid xy indices

        # Convert anchor indexes to int
        a = t[:, 6].long()

        # Add target tensors for this yolo layer to the output lists
        # Add to index list and limit index range to prevent out of bounds
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))

        # Add to target box list and convert box coordinates from global grid coordinates to local offsets in the grid cell
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box

        # Add correct anchor for each target to the list
        anchors_list.append(anchors[a])

        # Add class for each target to the list
        tcls.append(c)

    return tcls, tbox, indices, anchors_list

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        #vgg = vgg16(pretrained=True)
        #loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        #for param in loss_network.parameters():
        #    param.requires_grad = False
        #self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        #perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 2e-8 * tv_loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
