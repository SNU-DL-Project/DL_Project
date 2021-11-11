from __future__ import division

import os
import numpy as np

import torch
import torch.nn as nn

from utils.load import layer_block_config
from utils.layers import block_to_layer, YOLOLayer
from utils.utils import weights_init_normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Darknet(nn.Module):
    """
    1. DarkNet

    YoloV3의 최종 Network
    """

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.block_list = layer_block_config(config_path)
        self.hyperparams, self.module_list = block_to_layer(self.block_list)
        self.yolo_layers = [layer[0] for layer in self.module_list if isinstance(layer[0], YOLOLayer)]
        self.img_size = img_size

        # 이거는 데이터 로드/세이브할 때 필요하다.
        # 추후에 구현 예정
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x):
        img_size = x.size(2)
        layer_outputs, yolo_outputs = [], []

        for i, (blocks, module) in enumerate(zip(self.block_list[1:], self.module_list)):
            if blocks["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)

            elif blocks["type"] == "route":
                combined_outputs = torch.cat([layer_outputs[int(layer_i)] for layer_i in blocks["layers"].split(",")], 1)
                group_size = combined_outputs.shape[1]
                x = combined_outputs[:, 0: group_size]

            elif blocks["type"] == "shortcut":
                layer_i = int(blocks["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]

            elif blocks["type"] == "yolo":
                x = module[0](x, img_size)
                yolo_outputs.append(x)

            layer_outputs.append(x)

        # Inference와 Training을 구분하여야 한다.
        return yolo_outputs if self.training else torch.cat(yolo_outputs, 1)


        #################
        #   디버깅용!!    #
        #################
        #return yolo_outputs, torch.cat(yolo_outputs, 1)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        # If the weights file has a cutoff, we can find out about it by looking at the filename
        # examples: darknet53.conv.74 -> cutoff is 74
        filename = os.path.basename(weights_path)
        if ".conv." in filename:
            try:
                cutoff = int(filename.split(".")[-1])  # use last part of filename
            except ValueError:
                pass

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(
                    weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

def load_yolo_model(model_path, weights_path=None):
    """Loads the yolo model from file.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :return: Returns model
    :rtype: Darknet
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select device for inference
    model = Darknet(model_path).to(device)

    model.apply(weights_init_normal)

    # If pretrained weights are specified, start from checkpoint or weight file
    if weights_path:
        if weights_path.endswith(".pth"):
            # Load checkpoint weights
            model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            # Load darknet weights
            model.load_darknet_weights(weights_path)
    return model


#################
#   디버깅용!!    #
#################
if __name__ == "__main__":
    from torchsummary import summary

    path2config = "yolov3.cfg"
    model = Darknet(path2config).to(device)
    x = torch.rand(1, 3, 416, 416).to(device)
    with torch.no_grad():
        yolo_outputs, yolo_out_cat = model.forward(x)
        print(yolo_out_cat.shape)
        print(yolo_outputs[0].shape, yolo_outputs[1].shape, yolo_outputs[2].shape)

    summary(model, (3, 416, 416))