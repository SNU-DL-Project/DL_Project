import models
from utils.imple_utils import detect
from utils.imple_utils import _create_data_loader, _draw_and_save_output_images
from utils.evaluate import yolo_evaluate
from utils.dataloader import create_validation_data_loader,create_train_data_loader
from utils.imple_utils import _create_data_loader
from utils.evaluate import srres_evaluate, yolo_evaluate, print_yolo_eval_stats
import os
from utils.imple_utils import detect, detect_image, _draw_and_save_output_images_changed


#img save

# model_path = './config/yolov3.cfg'
# weights_path = './yolo_final.weight'
# img_path='./tmp_data/uav0000339_00001_v/'
# img_txtfile='./data/custom/images/video_path.txt'
# classes='/data/classes.names'
# output_path = './videooutput/labeled_img/'
#
# yoloModel = models.load_yolo_model(model_path, weights_path)

#detect_directory(model_path, weights_path, img_path, classes, output_path, batch_size=8, img_size=416, n_cpu=0, conf_thres=0.1, nms_thres=0.5)

#test_dataloader = _create_data_loader(img_path, batch_size=4, img_size=416, n_cpu=0)


#print(test_dataloader.dataset.)


# #img list 만들기
# f=open(img_txtfile,'r')
# li=f.read().strip()
# li_list = li.split('\n')
# if li_list[-1] =='':
#     del li_list[-1]
#

img_path = './data/custom_video/images/'
imple_dataloader = _create_data_loader(img_path, batch_size=4, img_size=416, n_cpu=0)
#모델 로드
model_path = './config/yolov3.cfg'
weights_path = './model/Yolo/yolo_final.weight'
yoloModel_imple = models.load_yolo_model(model_path, weights_path)

#metric 계산
img_detections, imgs = detect(yoloModel_imple, validation_dataloader2, output_path, img_size=416, conf_thres=0.1, nms_thres=0.5)

_draw_and_save_output_images_changed(img_detections, imgs, img_size, output_path, classes)

