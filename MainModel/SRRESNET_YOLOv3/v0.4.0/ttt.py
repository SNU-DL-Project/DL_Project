import random

import torch
from matplotlib.ticker import NullLocator
from matplotlib import patches

import models
from PIL import Image
import torchvision.transforms as transforms
import utils.yoloutils
import matplotlib.pyplot as plt
from utils.evaluate import non_max_suppression
import numpy as np
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
from utils.evaluate import yolo_evaluate
from utils.dataloader import ListDataset, DataLoader
from utils.dataloader import create_validation_data_loader
import os
from pytorchyolo.utils.utils import load_classes, rescale_boxes

trans=transforms.ToTensor()
# a=Image.open('./0000003.jpg')
# a=a.resize((416,416))

# img=YOLO_MODEL(torch.unsqueeze(trans(a),dim=0))
# Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
#
# YOLO_MODEL = models.load_yolo_model('./config/yolov3.cfg', './yolo_final.weight')
# # imgs = Variable(a.type(Tensor), requires_grad=False)
# # output = non_max_suppression(np.array(YOLO_MODEL(torch.unsqueeze(imgs,dim=0))),0.5,0.5)
# img_path = './video_path.txt'
# dataset = ListDataset(img_path, img_size=416, multiscale=False, transform='DEFAULT_TRANSFORMS')
# dataloader = DataLoader(
#     dataset,
#     batch_size=8,
#     shuffle=False,
#     num_workers=0,
#     pin_memory=True,
#     collate_fn=dataset.collate_fn)
#
# output = yolo_evaluate(YOLO_MODEL, dataloader,'./data/classes.names' , 416, 0.5, 0.1, 0.5, verbose=False)
#

li=os.listdir('./data/custom_yolo/images/')

f=open('./data/valid_Yolo.txt','w')
for i, data in enumerate(li):
    f.write('./data/custom_yolo/images/'+data +'\n')
f.close()






def detect_image(model, image, img_size=416, conf_thres=0.5, nms_thres=0.5):
    """Inferences one image with model.
    :param model: Model for inference
    :type model: models.Darknet
    :param image: Image to inference
    :type image: nd.array
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: Detections on image with each detection in the format: [x1, y1, x2, y2, confidence, class]
    :rtype: nd.array
    """
    model.eval()  # Set model to evaluation mode

    # Configure input
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(img_size)])(
        (image, np.zeros((1, 5))))[0].unsqueeze(0)

    if torch.cuda.is_available():
        input_img = input_img.to("cuda")

    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], img_size, image.shape[:2])
    return detections



#
# a=Image.open('./0000003.jpg')
# a=a.resize((416,416))
#
#
# YOLO_MODEL = models.load_yolo_model('./config/yolov3.cfg', './yolo_final.weight')
# img=trans(a)
# detection=detect_image(YOLO_MODEL, np.array(a))
#
# _draw_and_save_output_image('0000003.jpg',detection, 416,'./tmp_data/','./data/classes.names')
#
# ####################################################################################################################################

path='./data/custom_yolo/labels/'
l=os.listdir(path)

for i,d in enumerate(l):
    with open(path+l[i], "r") as f:
        lines = f.readlines()
    with open(path+l[i], "w") as f:
        for line in lines:
            print(line[0])
            if int(line.split(' ')[0]) <= 5:
                f.write(line)

