import random
import torchvision.utils as U
import os
from bounding_box import bounding_box as bb
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
trans = transforms.ToTensor()

# 10000장의 각각에대한 이미지 list  /  10000개에 대한 각각의 dir가 담긴 txt_label_origin, txt_label_inference
def Yolo(img_origin, img_inference, txt_label_origin, txt_label_inference, row_nums):

    #dictionary
    label_dic = { 0 : "pedestrian", 1 : "car", 2 : "van", 3 : "truck", 4 : "bus",
                  5 : "motor"}
    color_dic = { 0 : "orange", 1 : "yellowgreen", 2 : "cyan", 3 : "pink", 4 : "white",
                  5 : "purple"}
    if not os.path.exists(os.getcwd()+'/result'):
        os.makedirs(os.getcwd()+'/result')
    if not os.path.exists(os.getcwd()+'/result/Yolo'):
        os.makedirs(os.getcwd()+'/result/Yolo')
        os.makedirs(os.getcwd()+'/result/Yolo/imgs/origin')
        os.makedirs(os.getcwd()+'/result/Yolo/imgs/inference')
        os.makedirs(os.getcwd()+'/result/Yolo/metrics')

    #10000장에 대해 iteration,
    '''
    각 장 마다 해야하는 행동
    1. bbox 그리기 (score넣기?)
    2. 저장하기 
    3. print 하기
    '''
    img_list_tmp =[]

    for i in range(0, len(img_origin)):
        #1
        ##origin
        label_origin=read_txt(txt_label_origin[i])
        img_origin[i]=img_origin[i].numpy().transpose(2,1,0)
        for j in range(0,len(label_origin)):
            x1,y1,x2,y2 = xywh2minmax(label_origin[j][1], label_origin[j][2], label_origin[j][3], label_origin[j][4])
            bb.add(img_origin[i], x1, y1, x2, y2, None, color_dic[int(label_origin[j][0])]) # 샐깔
        #label_dic[int(label_origin[i][0])] -> class명

        ##inference
        label_inference = read_txt(txt_label_inference[i])
        img_inference[i]=img_inference[i].numpy().transpose(2,1,0)
        for j in range(0,len(label_inference)):
            x1,y1,x2,y2 = xywh2minmax(label_inference[j][1], label_inference[j][2], label_inference[j][3], label_inference[j][4])
            bb.add(img_inference[i], x1, y1, x2, y2, None,color_dic[int(label_inference[j][0])]) # 샐깔
        #label_dic[int(label_origin[i][0])] -> class명
        #2
        name = txt_label_origin[i].split('/')[-1].split('.')[0] #############################################################이부분은 구현을 어떻게 하냐에 따라 수정필요, 이름만 따오고 싶은 것
        U.save_image(trans(img_inference[i]), os.getcwd() + '/result/Yolo/imgs/inference/' + name + '_inference.jpg')
        U.save_image(trans(img_origin[i]), os.getcwd() + '/result/Yolo/imgs/origin/' + name + '_origin.jpg')

    #3
    rs = random.sample(range(0,len(img_origin)),row_nums) #rs뽑고
    for i in range(0, len(rs)): #총 10000번동인
        img_list_tmp.append(img_origin[rs[i]])
        img_list_tmp.append(img_inference[rs[i]])

    showpic(img_list_tmp, row_nums)


def read_txt(dir):
    f= open(dir, 'r')
    txt = f.read()
    line_list = txt.split('\n')

    if line_list[-1] == '':
        del line_list[-1]
    result_list=[]
    for i in range(0,len(line_list)):

        word_list = line_list[i].split(' ')
        #print(word_list)
        result_list.append(word_list)

    return result_list

def xywh2minmax(x,y,w,h):

    x1=int(416* (float(x) - 0.5 * float(w))) # xmin
    x2=int(416 * (float(x) + 0.5 * float(w))) # xmax
    y1=int(416 * (float(y) - 0.5 * float(h))) # ymin
    y2=int(416 * (float(y) + 0.5 * float(h))) # ymax

    return x1, y1, x2, y2

#images list는 원본과 infernce순서가 concat돼서 계속 더해진것
def showpic(images, row_num): #type(img)=list of 3D tensor [C,W,H]
    rows=row_num; cols=2; imgsize=15;
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(imgsize, imgsize))

    colors=["orange",  "yellowgreen", "cyan",  "pink",  "white", "purple"]
    patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=colors[i]) ) for i in range(len(colors)) ]

    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=6, borderaxespad=0. )

    for ind,img in enumerate(images):

        if ind%2==0:
            title= 'Origin'

        else :
            title= 'Inference'
            # ax.ravel()[ind].text(5,50,'SSIM: '+str(round(ssim[int(ind/3)],3))+'\n'+'PSNR: '+str(round(psnr[int(ind/3)], 2)),
            #                      ha='center',va='bottom',bbox={'facecolor' : 'white'},size='smaller')
        img = torch.from_numpy(img)
        ax.ravel()[ind].imshow(img)
        ax.ravel()[ind].set_title(title)

        ax.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.show()
