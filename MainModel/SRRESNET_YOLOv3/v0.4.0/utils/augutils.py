import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from PIL import Image, ImageFilter

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from .yoloutils import xywh2xyxy_np

def sr_downsample(imgdata, noise=True, down_size=104): # type(img) = torch.Tensor
    #noise filter는 따로 내장함수가 없더군요. 착각했습니다. 원하시는 만큼 mean var값 변경시켜가면서 noise 정도를 정하시면 됩니다.
    #전자는 gaussian 정규분포 노이즈입니다. 후자(salt and pepper)는 흑백 노이즈입니다.
    toPil=transforms.ToPILImage()
    toTen=transforms.ToPILImage()
    imgdatalist=[]
    batch_size = imgdata.size(0)
    for i in range(0, batch_size):
        imgdatalist.append(imgdata[i])
    def noisy(image,noise_typ):
        if noise_typ == "gauss":
            row,col,ch= image.shape
            mean = 0
            var = 0.5
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            return noisy
        elif noise_typ == "s&p":
            row,col,ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0
            return out


    #final_img는 Pillow type입니다. dataloader의 dataset(=imgdata)은 tensor형태일것이니, tensor->numpy(->pillow 작업이 필요힙니다.

    imgdatalist_updated=[]
    tmp=[]
    for i in range(0,batch_size):
        get_img=toPil(imgdatalist[i])
        if noise:
            #변환작업
            #TODO Blur값 조정 필요!!
            blurImage = get_img.filter(ImageFilter.GaussianBlur(1.0))
            downImage = blurImage.resize((down_size, down_size)) #size=(416,416) or something
            #print(noisy(np.array(downImage), 'gauss').T.shape) 디버깅
            noiseImage = noisy(np.array(downImage), 'gauss') / 255.0
            noiseImage = noiseImage.swapaxes(1, 2)
            noiseImage = noiseImage.swapaxes(0, 1)
            noiseImage = torch.from_numpy(noiseImage)
            #output인 noiseimage는 numpy 형태이므로 다시 tensor로 바꿔주어야합니다.
        else:
            downImage = get_img.resize((down_size, down_size))  # size=(416,416) or something
            noiseImage = np.array(downImage) / 255.0
            noiseImage = noiseImage.swapaxes(1, 2)
            noiseImage = noiseImage.swapaxes(0, 1)
            noiseImage = torch.from_numpy(noiseImage)

        tmp.append(noiseImage)


    '''
    blur종류는 다음과 같습니다. BLUR, Box BLUR, Gaussian BLUR, 블러 정도를 조절하고싶다면 다음을 참고하세요
    documentation : https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html
    noise 종류는 gauss를 하시던지 s&p를 쓰시던지 아무거나 쓰시면 됩니다. 아무래도 색필터가 더 낫지않을까 싶긴합니다
    '''

    result = torch.stack(tmp, dim=0)
    return result.float()

class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        # Unpack data
        img, boxes = data

        # Convert xywh to xyxy
        boxes = np.array(boxes)
        boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])

        # Convert bounding boxes to imgaug
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*box[1:], label=box[0]) for box in boxes],
            shape=img.shape)

        # Apply augmentations
        img, bounding_boxes = self.augmentations(
            image=img,
            bounding_boxes=bounding_boxes)

        # Clip out of image boxes
        bounding_boxes = bounding_boxes.clip_out_of_image()

        # Convert bounding boxes back to numpy
        boxes = np.zeros((len(bounding_boxes), 5))
        for box_idx, box in enumerate(bounding_boxes):
            # Extract coordinates for unpadded + unscaled image
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            # Returns (x, y, w, h)
            boxes[box_idx, 0] = box.label
            boxes[box_idx, 1] = ((x1 + x2) / 2)
            boxes[box_idx, 2] = ((y1 + y2) / 2)
            boxes[box_idx, 3] = (x2 - x1)
            boxes[box_idx, 4] = (y2 - y1)

        return img, boxes

class RelativeLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] /= w
        boxes[:, [2, 4]] /= h
        return img, boxes

class AbsoluteLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] *= w
        boxes[:, [2, 4]] *= h
        return img, boxes

class PadSquare(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(
                1.0,
                position="center-center").to_deterministic()
        ])

class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)

        bb_targets = torch.zeros((len(boxes), 6))
        bb_targets[:, 1:] = transforms.ToTensor()(boxes)

        return img, bb_targets

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, boxes = data
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return img, boxes

class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-10, 10)),
            iaa.Fliplr(0.5),
        ])

class StrongAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-20, 20)),
            iaa.Fliplr(0.5),
        ])

AUGMENTATION_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    StrongAug(),
    #DefaultAug(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])

DEFAULT_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])