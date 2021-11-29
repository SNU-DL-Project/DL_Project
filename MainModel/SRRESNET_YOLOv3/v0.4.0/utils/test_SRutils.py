import matplotlib.pyplot as plt
import os
import torchvision.utils as U
import random
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def showpic(images, psnr, ssim, row_num): #type(img)=list of 3D tensor [C,W,H]
    rows=row_num; cols=3; imgsize=10;
    figure, ax = plt.subplots(nrows=rows,ncols=cols,figsize=(10, 10))


    for ind,img in enumerate(images):

        if ind%3==0:
            title= 'Low'
        elif ind%3==1:
            title= 'High'
        else :
            title= 'Origin'
            print(f'val : {int(ind/3)}')
            ax.ravel()[ind].text(5,50,'SSIM: '+str(round(ssim[int(ind/3)],3))+'\n'+'PSNR: '+str(round(psnr[int(ind/3)], 2)),
                                 ha='center',va='bottom',bbox={'facecolor' : 'white'},size='smaller')

        ax.ravel()[ind].imshow(img.permute(1,2,0))
        ax.ravel()[ind].set_title(title)

        ax.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.show()


def get_ssim(img_H, img_O):
    img_H=np.array(img_H)
    img_O=np.array(img_O)
    ssim_value = ssim(img_H, img_O, data_range=img_H.max() - img_H.min(),multichannel=True)
    return ssim_value


def get_psnr(ori_img, con_img):
    ori_img = ori_img.numpy()
    con_img = con_img.numpy()
    # # 해당 이미지의 최대값 (채널 최대값 - 최솟값)
    # max_pixel = 255.0
    # # MSE 계산
    # mse = np.mean((ori_img - con_img)**2)
    # if mse ==0:
    #     return 100
    # # PSNR 계산
    # psnr = 20* math.log10(max_pixel / math.sqrt(mse))

    return psnr(ori_img, con_img, data_range=con_img.max()-con_img.min())



##input img= list of 3D tensor [C,W,H]
#img1 = [104*104, ...]
#img2 = [416*416, 원본이 아닌 SR사진, ...]
#img3 = [원본, ...]
#name_L = [사진1name, 사진2name, ...]

#Low, High, Original



def SR(img_L, img_H, img_O, imgname,row_num):
    #예시로 row_num개만 출력
    ## print ##############################################################################


    idx_list=[]
    psnr_list=[]
    ssim_list=[]
    img_list=[]

    # 2개만 출력, random하게 추출
    for i in range(0,row_num):
        idx_list=random.sample(range(0, len(img_L)), len(img_L)) # 중복업이 뽑음

    for i in idx_list:
        # permute 하는 이유는 PIL to tensor 과정에서 (a,b,c)->(c,b,a) 가 된다.
        psnr = get_psnr(img_O[i].permute(2, 1, 0), img_H[i].permute(2, 1, 0))
        ssim = get_ssim(img_O[i].permute(2, 1, 0), img_H[i].permute(2, 1, 0))
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        #list생성 LHO 순서대로 그냥 쌓기


        img_list.append(img_L[i])
        img_list.append(img_H[i])
        img_list.append(img_O[i])

    showpic(img_list,psnr_list, ssim_list, row_num)
    ####################################################################################

    psnr_list_tmp=[]
    ssim_list_tmp=[]
    ## save ##############################################################################
    #makedir
    if not os.path.exists(os.getcwd()+'/result'):
        os.makedirs(os.getcwd()+'/result')
    if not os.path.exists(os.getcwd()+'/result/SR'):
        os.makedirs(os.getcwd()+'/result/SR')
        os.makedirs(os.getcwd()+'/result/SR/imgs')
        os.makedirs(os.getcwd()+'/result/SR/metrics')

    for i in range(0, len(img_L)):

        #img save
        U.save_image(img_L[i], os.getcwd() + '/result/SR/imgs/'+ imgname[i] + '_Lr.jpg')
        U.save_image(img_H[i], os.getcwd() + '/result/SR/imgs/'+ imgname[i] + '_Hr.jpg')
        U.save_image(img_O[i] , os.getcwd() + '/result/SR/imgs/'+ imgname[i] + '_Or.jpg')

        psnr = get_psnr(img_H[i].permute(2, 1, 0), img_O[i].permute(2, 1, 0))
        ssim = get_ssim(img_H[i].permute(2, 1, 0), img_O[i].permute(2, 1, 0))

        psnr_list_tmp.append(psnr)
        ssim_list_tmp.append(ssim)

        #psnr save
        if i==0: f=open('result/SR/metrics/psnr.txt','w')
        else : f=open('result/SR/metrics/psnr.txt','a')
        f.write(imgname[i]+'_psnr: '+ str(psnr) + '\n')
        f.close()

        #ssim save
        if i == 0 : f1=open('result/SR/metrics/ssim.txt', 'w')
        else : f1=open('result/SR/metrics/ssim.txt', 'a')
        f1.write(imgname[i]+'_ssim: '+ str(ssim) + '\n')
        f1.close()

    ##statistic save
    f2=open('result/SR/metrics/statistic.txt', 'w')
    f2.write('psnr_mean: '+ str(np.mean(psnr_list_tmp))+'\n')
    f2.write('ssim_mean: '+ str(np.mean(ssim_list_tmp))+'\n')
    f2.close()
    #####################################################################################

    return;