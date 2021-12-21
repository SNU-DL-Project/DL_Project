import torchvision.utils as U
import os
import numpy as np
from PIL import Image
import utils.bounding_box as bb
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
trans = transforms.ToTensor()

# 10000장의 각각에대한 이미지 list  /  10000개에 대한 각각의 dir가 담긴 txt_label_origin, txt_label_inference
def Yolo(img_origin, img_inference, txt_label_origin, row_nums, wantlabel=False):

    #dictionary
    label_dic = { 0 : "pedestrian", 1 : "car", 2 : "van", 3 : "truck", 4 : "bus",
                  5 : "motor"}
    color_dic = { 0 : (51,153,255), 1 : (102,255,178), 2 : (255,255,51), 3 : (255,0,255), 4 : (255,255,255),
                  5 : (204,0,102)}

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
    4. metric 저장 및 출력
    '''
    img_list_tmp =[]
    img_origin_new=[]
    for i in range(0, len(img_origin)):
        #1
        ##origin bbox drawing
        label_origin=read_txt(txt_label_origin[i])

        img_origin_new.append(np.array(Image.open(img_origin[i]).resize((416,416))))


        for j in range(0,len(label_origin)): #하나의 txt파일안을 돌면서
            if wantlabel == False or label_dic[int(label_origin[j][0])] in wantlabel:
                x1,y1,x2,y2 = xywh2minmax(label_origin[j][1], label_origin[j][2], label_origin[j][3], label_origin[j][4])
                bb.add(img_origin_new[i], x1, y1, x2, y2, None, color_dic[int(label_origin[j][0])]) # 색깔
        #label_dic[int(label_origin[i][0])] -> class명

    # ##inference bbox drawing
    # label_inference = read_txt(txt_label_inference[i])
    # img_inference[i]=img_inference[i].numpy().transpose(2,1,0)
    # for j in range(0,len(label_inference)):
    #     if wantlabel == False or label_dic[int(label_inference[j][0])] in wantlabel:
    #         x1,y1,x2,y2 = xywh2minmax(label_inference[j][1], label_inference[j][2], label_inference[j][3], label_inference[j][4])
    #         bb.add(img_inference[i], x1, y1, x2, y2, None,color_dic[int(label_inference[j][0])]) # 색깔
    # #label_dic[int(label_inference[i][0])] -> class명

        #2
        name = txt_label_origin[i].split('/')[-1].split('.')[0] ##################이부분은 구현을 어떻게 하냐에 따라 수정필요, 이름만 따오고 싶은 것
        #U.save_image(trans(img_inference[i]), os.getcwd() + '/result/Yolo/imgs/inference/' + name + '_Inference.jpg')
        U.save_image(trans(img_origin_new[i]), os.getcwd() + '/result/Yolo/origin_labeled_img/' + name + '_Origin.jpg')

    #3
    # rs = random.sample(range(0,len(img_origin_new)), row_nums) #rs뽑고
    for i in range(0, len(img_origin_new)): #총 10000번동인
        img_list_tmp.append(img_origin_new[i])
        img_list_tmp.append(img_inference[i])

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
        result_list.append(word_list)

    return result_list

def xywh2minmax(x,y,w,h):

    x1=int(416* (float(x) - 0.5 * float(w))) # xmin
    x2=int(416 * (float(x) + 0.5 * float(w))) # xmax
    y1=int(416 * (float(y) - 0.5 * float(h))) # ymin
    y2=int(416 * (float(y) + 0.5 * float(h))) # ymax

    return x1, y1, x2, y2

#images list는 원본과 infernce 순서로 concat돼서 계속 더해진것
def showpic(images2, row_num): #type(img)=list of 3D tensor [C,W,H]
    rows=row_num; cols=2; imgsize=15;
    figure, ax = plt.subplots(nrows=rows+1, ncols=cols, figsize=(imgsize, imgsize))

    colors=["orange",  "yellowgreen", "cyan",  "pink",  "white", "purple"]
    labels=["pedestrian",  "car", "van",  "truck",  "bus", "motor"]
    patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=labels[i]) ) for i in range(len(colors)) ]

    #여유를 위해 많이만들어 놓음
    xlabels = ["origin", "inference","origin", "inference","origin", "inference","origin", "inference","origin", "inference"]
    for i, img in enumerate(images2):

        roww= row_num;coll=2;
        ax = plt.subplot(roww, coll, i+1)
        ax.imshow(img)
        ax.set_xlabel(xlabels[i])
        ax.set_xticks([]), ax.set_yticks([])
    plt.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=6, borderaxespad=1 )

    plt.show()



def giveMeImageCollection(path):
    f = open(path, 'r')
    lis=f.read()
    lis_list = lis.split('\n')
    if lis_list[-1] =='':
        del lis_list[-1]
    return lis_list