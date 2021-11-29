from PIL import Image
import numpy as np
import utilstorchvision.transforms as transforms
from utils.test_Yolov3utils import Yolo
from utils.test_SRutils import SR
import os
import torch

if __name__=="__main__":
    ################### debug #############################
    trans = transforms.ToTensor()
    ##
    img1=Image.open('iky2.jpg')
    w,h =img1.size
    img1_O=img1; img1_L=img1.resize((int(w/20), int(h/20))); img1_H=img1_L.resize((w, h))
    img1_O=trans(img1_O); img1_L=trans(img1_L); img1_H=trans(img1_H)
    ##
    img2=Image.open('iky3.jpg')
    w,h =img2.size
    img2_O=img2; img2_L=img2.resize((int(w/200), int(h/100))); img2_H=img2_L.resize((w, h))
    img2_O=trans(img2_O); img2_L=trans(img2_L); img2_H=trans(img2_H)
    ##
    img3=Image.open('iky4.jpg')
    w,h =img3.size
    img3_O=img3; img3_L=img3.resize((int(w/200), int(h/100))); img3_H=img3_L.resize((w, h))
    img3_O=trans(img3_O); img3_L=trans(img3_L); img3_H=trans(img3_H)
    ##
    img4=Image.open('iky5.jpg')
    w,h =img4.size
    img4_O=img4; img4_L=img4.resize((int(w/200), int(h/100))); img4_H=img4_L.resize((w, h))
    img4_O=trans(img4_O); img4_L=trans(img4_L); img4_H=trans(img4_H)
    ##
    img_O_list=[img1_O, img2_O, img3_O, img4_O]
    img_L_list=[img1_L, img2_L, img3_L, img4_L]
    img_H_list=[img1_H, img2_H, img3_H, img4_H]


    ###
    SR(img_L_list, img_H_list, img_O_list, ['iky2','iky3', 'iky4','iky5'], row_num=len(img_H_list))
    ###
    ################### debug #############################



    ##############

    #1. test dataload 해오기
    #2. infernce시키기

    ##############


    # 10000장의 각각에대한 이미지 list  /  10000개에 대한 각각의 dir가 담긴 txt_label_origin, txt_label_inference
    ori_list=[]
    inf_list=[]
    txt_o_list=[]
    txt_i_list=[]
    file_list = os.listdir('./test')
    for i, data in enumerate(file_list):
        if data.split('.')[-1] == 'jpg':
            img_origin=Image.open('./test/'+data)

            img_origin=np.array(img_origin).transpose(2,1,0)
            img_origin=torch.from_numpy(img_origin)
            ori_list.append(img_origin)
            inf_list.append(img_origin)

        if data.split('.')[-1] == 'txt':
            txt_o_list.append('./test/'+data)
            txt_i_list.append('./test/'+data)

    # print(ori_list)
    # print(txt_o_list)

    img_origin=ori_list
    img_inference=inf_list
    txt_label_origin=txt_o_list
    txt_label_inference=txt_i_list

    #input은 tensorlist여야함 #10000개에 대한 각각의 dir가 담긴 txt_label_origin, txt_label_inference
    Yolo(img_origin, img_inference, txt_label_origin, txt_label_inference,row_nums=2)


