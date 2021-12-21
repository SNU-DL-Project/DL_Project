from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from utils.test_Yolov3utils import Yolo
from utils.test_SRutils import SR
from utils.dataloader import create_validation_data_loader
import os
import torch
from utils.imple_utils import _create_data_loader
from models import load_srres_model
from utils.evaluate import srres_evaluate, yolo_evaluate, print_yolo_eval_stats
import os
from utils.imple_utils import detect, _draw_and_save_output_images_for_metric
from models import load_yolo_model
from utils.test_Yolov3utils import giveMeImageCollection

os.environ['KMP_DUPLICATE_LIB_OK']='True'
trans = transforms.ToTensor()

#
#
#
# # SRresnet ##
# # start  ########################################################################################################################
#
# #이미지 경로
# img_path = './data/valid_SR.txt'
# #data 생성
# batch_size=1
# SR_dataloader =create_validation_data_loader(img_path, batch_size=batch_size, img_size=416, n_cpu=0)
# #모델로드
# SRmodel = load_srres_model(model_path_srres='./config/srres_16.cfg', weights_path_srres='./model/SRres/srres_ckpt_74.pth')
# #metric 계산
# metrics_output, SR_img_list, Origin_img_list, low_img_list, psnr_lists, mse_lists, ssim_lists = \
#     srres_evaluate(SRmodel, SR_dataloader, verbose=False)
#
# Origin_img_list_new=[]; low_list_new=[];SR_img_list_new=[]
#
# # test 하기위해서는 batch size가 1 이어야 한다. (metric calculation)
# if batch_size == 1 :
#     Origin_img_list_new = Origin_img_list
#     low_list_new = low_img_list
#     SR_img_list_new = SR_img_list
# else: # img batch parsing
#     for i in range(0,(len(low_img_list))):
#         Origin_img_list_new.append(Origin_img_list[0][i])
#         low_list_new.append(low_img_list[0][i])
#         SR_img_list_new.append(SR_img_list[0][i])
#
# '''
# #########################################
# # SR() : pop up images and save metrics #
# #########################################
#
# Input SPEC
# 1. img_L_list           : Low resolution의 test_img가 담긴 list
# 2. img_S_list           : Super resolution의 test_img가 담긴 list
# 4. img_O_list           : Origin resolution의 test_img가 담긴 list
# 5. imgname              : img들의 name 순서대로 담은 list / 이미지저장할때 필요
# 6. row_num              : 몇 줄을 출력하고 싶은지
# '''
#
# SR(low_list_new, SR_img_list_new, Origin_img_list_new, imgname=list(map(str,np.arange(len(low_list_new)))), row_num=2, batch_size = batch_size, ssim_lists_new=ssim_lists, psnr_lists_new=psnr_lists, mse_lists_new=mse_lists)
#
# # end  ########################################################################################################################














# parameters
img_txtfile='./data/valid_Yolo.txt'
classes=["pedestrian", "car", "van", "truck","bus" ,"motor"]
output_path = './result/Yolo/images/'
path_list='data/custom_yolo/images/'
img_path2 = './data/valid_Yolo.txt'
HRmodelpath = './model/Yolo/HR/HRyolo_final.weight'
LRmodelpath = './model/Yolo/LR/LRyolo_final.weight'

# dataloader 생성
metric_dataloader =create_validation_data_loader(img_path2, batch_size=4, img_size=416, n_cpu=0) # metric용
imageSave_dataloader=_create_data_loader(path_list, batch_size=4, img_size=416, n_cpu=0) # 사진저장용

# 모델로드
Yolomodel = load_yolo_model(model_path='./config/yolov3.cfg', weights_path = HRmodelpath)

# metric 계산
wantTolabel_infer=['pedestrian', 'car', 'van', 'bus', 'motor', 'truck']
img_detections, imgs = detect(Yolomodel, imageSave_dataloader, output_path, img_size=416, conf_thres=0.1, nms_thres=0.5)

# inference하기
_draw_and_save_output_images_for_metric(img_detections, imgs, img_size=416, output_path=output_path, classes=classes)

##metric 뽑아내기
metrics_output_yolo = yolo_evaluate(Yolomodel, metric_dataloader, classes, img_size=416, iou_thres=0.5, conf_thres=0.1, nms_thres=0.5, verbose=False)
precision, recall, AP, F1, ap_class = metrics_output_yolo

# txt에 metrcis save
f1=open('result/Yolo/metrics/statistics.txt','w')
precision_new=[]; recall_new=[]; AP_new=[]; F1_new=[];
for i in range(0,6):
    precision_new.append('{:.8f}'.format(round(precision[i],8)))
    recall_new.append('{:.8f}'.format(round(recall[i],8)))
    AP_new.append('{:.8f}'.format(round(AP[i],8)))
    F1_new.append('{:.8f}'.format(round(F1[i],8)))

f1.write(f'              pedestrian         car             van            truck            bus            motor     '+'\n')
f1.write(f'precision : {precision_new[0]} {precision_new[1]}  {precision_new[2]}  {precision_new[3]}  {precision_new[4]}  {precision_new[5]} '+'\n')
f1.write(f'recall      : {recall_new[0]} {recall_new[1]}  {recall_new[2]}  {recall_new[3]}  {recall_new[4]}  {recall_new[5]} '+'\n')
f1.write(f'AP         : {AP_new[0]} {AP_new[1]}  {AP_new[2]}  {AP_new[3]}  {AP_new[4]}  {AP_new[5]} '+'\n')
f1.write(f'F1 score   : {F1_new[0]} {F1_new[1]}  {F1_new[2]}  {F1_new[3]}  {F1_new[4]}  {F1_new[5]} '+'\n')
f1.write('\nmAP : '+ str(sum(list(map(float,AP_new)))/len(AP_new)))
f1.close()


#print table on console
print_yolo_eval_stats(metrics_output=metrics_output_yolo, class_names=classes, verbose=True)

#내가 보고싶은 문서(원본) -> ./data/custom_yolo/wantToSee.txt 가서 보고 싶은 사진의 경로를 추가함으로써 txt 내용 수정할것.
img_origin = giveMeImageCollection(path = './data/custom_yolo/wantToSee.txt')


listdirec = os.listdir(path = './result/Yolo/images/')
img_inference=[]
for i, data in enumerate(listdirec):
    img_inference.append(trans(Image.open('./result/Yolo/images/'+data).resize((416,416))).permute(1,2,0))


#origin img txt 라벨 준비
tmp = os.listdir('./data/custom_yolo/labels/')
txt_list=[]
for i in range(0, len(tmp)):
    txt_list.append('./data/custom_yolo/labels/'+tmp[i])

'''
###################################
# Yolo : pop up images and save metrics #
###################################
Input SPEC
1. img_origin           : origin_img의 tensor type list ex)10000장
2. img_inference        : inference_img의 tensor type list ex)10000장
3. txt_label_origin     : orogin_img의 labeling_txt 경로가 담긴 list
4. txt_label_inference  : inference_img의 labeling_txt 경로가 담긴 list
5. row_nums             : 몇줄을 출력하고 싶은지
6. wantlabel            : boundingbox를 그리고 싶은 class명이 담긴 리스트
ex) ['pedestrian', 'car',  'van', 'bus', 'motor', 'truck']
'''
Yolo(img_origin, img_inference, txt_list, row_nums=len(img_origin), wantlabel=["pedestrian", "car", "van", "truck","bus" ,"motor"])



# #img path list 만들기
# f=open(img_txtfile,'r')
# li=f.read().strip()
# li_list = li.split('\n')
# if li_list[-1] =='':
#     del li_list[-1]
