import models
from utils.imple_utils import detect
from utils.imple_utils import _create_data_loader, _draw_and_save_output_images
from utils.evaluate import yolo_evaluate
from utils.dataloader import create_validation_data_loader,create_train_data_loader
from utils.imple_utils import _create_data_loader
from utils.evaluate import srres_evaluate, yolo_evaluate, print_yolo_eval_stats
import os
from utils.imple_utils import detect, detect_image, _draw_and_save_output_images_changed
import cv2

# #img list 만들기
# f=open(img_txtfile,'r')
# li=f.read().strip()
# li_list = li.split('\n')
# if li_list[-1] =='':
#     del li_list[-1]
#

# 데이터 만들기
img_path = './data/custom_video/images/'
imple_dataloader = create_validation_data_loader(img_path, batch_size=4, img_size=416, n_cpu=0)
#모델 로드
model_path = './config/yolov3.cfg'
weights_path = './model/Yolo/yolo_final.weight' #만들어주세요!
yoloModel_imple = models.load_yolo_model(model_path, weights_path)

#metric 계산
output_path = './videooutput/labeled_img/'
img_detections, imgs = detect(yoloModel_imple, imple_dataloader, output_path, img_size=416, conf_thres=0.1, nms_thres=0.5)

#img 저장
classes=["pedestrian", "car", "van", "truck","bus" ,"motor"]
_draw_and_save_output_images_changed(img_detections, imgs, img_size=416, output_path=output_path, classes=classes)

#동영상만들기
# out = './result/Yolo_video/'
# writer = None
# if writer is None:
#     # initialize our video writer
#     fourcc = cv2.VideoWriter_fourcc(*"DIVX")
#     writer = cv2.VideoWriter(out, fourcc, 30,
#                              (416, 416), True)
#
#     # write the output frame to disk
# writer.write(frame)
# # release the file pointers
# print("[INFO] cleaning up...")
# writer.release()
# vs.release()