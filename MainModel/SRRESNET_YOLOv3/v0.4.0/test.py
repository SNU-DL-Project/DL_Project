from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from utils.test_Yolov3utils import Yolo
from utils.test_SRutils import SR
import os
import torch


trans = transforms.ToTensor()
##

img1=Image.open('tmp_data/iky2.jpg')
w,h =img1.size
img1_O=img1; img1_L=img1.resize((int(w/20), int(h/20))); img1_S=img1_L.resize((w, h))
img1_O=trans(img1_O); img1_L=trans(img1_L); img1_S=trans(img1_S)
##
img2=Image.open('tmp_data/iky3.jpg')
w,h =img2.size
img2_O=img2; img2_L=img2.resize((int(w/200), int(h/100))); img2_S=img2_L.resize((w, h))
img2_O=trans(img2_O); img2_L=trans(img2_L); img2_S=trans(img2_S)
##
img3=Image.open('tmp_data/iky4.jpg')
w,h =img3.size
img3_O=img3; img3_L=img3.resize((int(w/200), int(h/100))); img3_S=img3_L.resize((w, h))
img3_O=trans(img3_O); img3_L=trans(img3_L); img3_S=trans(img3_S)
##
img4=Image.open('tmp_data/iky5.jpg')
w,h =img4.size
img4_O=img4; img4_L=img4.resize((int(w/200), int(h/100))); img4_S=img4_L.resize((w, h))
img4_O=trans(img4_O); img4_L=trans(img4_L); img4_S=trans(img4_S)
##
img_O_list=[img1_O, img2_O, img3_O, img4_O]
img_L_list=[img1_L, img2_L, img3_L, img4_L]
img_S_list=[img1_S, img2_S, img3_S, img4_S]

########################################################################################################################

ori_list=[]
inf_list=[]
txt_o_list=[]
txt_i_list=[]
file_list = os.listdir('tmp_data//test')
for i, data in enumerate(file_list):
    if data.split('.')[-1] == 'jpg':
        img_origin=Image.open('tmp_data//test/'+data)

        img_origin=np.array(img_origin).transpose(2,1,0)
        img_origin=torch.from_numpy(img_origin)
        ori_list.append(img_origin)
        inf_list.append(img_origin)

    if data.split('.')[-1] == 'txt':
        txt_o_list.append('tmp_data//test/'+data)
        txt_i_list.append('tmp_data//test/'+data)

## data준비
img_origin=ori_list
img_inference=inf_list
txt_label_origin=txt_o_list
txt_label_inference=txt_i_list


########################################################################################################################




#TODO 1,2
#1. test dataload 해오기
#2. infernce시킨 후 feature 뽑아내기
##############
# start





# end
###############









'''
#################################
# SR : pop up images and metric #
#################################

Input SPEC 
1. img_L_list           : Low resolution의 test_img가 담긴 list
2. img_S_list           : Super resolution의 test_img가 담긴 list
4. img_O_list           : Origin resolution의 test_img가 담긴 list
5. imgname              : img들의 name 순서대로 담은 list / 이미지저장할때 필요
6. row_num              : 몇줄을 출력하고 싶은지
'''



SR(img_L_list, img_S_list, img_O_list, imgname=['iky2','iky3', 'iky4','iky5'], row_num=2)


'''
###################################
# Yolo : pop up images and metric #
###################################
Input SPEC 
1. img_origin           : origin_img의 tensor type list ex)10000장
2. img_inference        : inference_img의 tensor type list ex)10000장
3. txt_label_origin     : orogin_img의 labeling_txt 경로가 담긴 list 
4. txt_label_inference  : inference_img의 labeling_txt 경로가 담긴 list
5. row_nums             : 몇줄을 출력하고 싶은지
6. wantlabel            : boundingbox를 그리고 싶은 class명이 담긴 리스트
ex) ['car', 'pedestrian', 'van', 'bus', 'motor', 'truck']
'''

Yolo(img_origin, img_inference, txt_label_origin, txt_label_inference, row_nums=2, wantlabel=['car', 'pedestrian'])


