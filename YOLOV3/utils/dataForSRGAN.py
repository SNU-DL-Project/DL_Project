from PIL import Image, ImageFilter
import numpy as np
import cv2
import os
import torch

def dataForSGgan(imgdata): # type(img) = torch.Tensor
    #noise filter는 따로 내장함수가 없더군요. 착각했습니다. 원하시는 만큼 mean var값 변경시켜가면서 noise 정도를 정하시면 됩니다.
    #전자는 gaussian 정규분포 노이즈입니다. 후자(salt and pepper)는 흑백 노이즈입니다.
    def noisy(image,noise_typ):
        if noise_typ == "gauss":
            row,col,ch= image.shape
            mean = 0
            var = 0.1
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
    get_img=(Image.fromarray(imgdata.numpy()))

    #변환작업
    blurImage = get_img.filter(ImageFilter.BLUR)
    downImage =  blurImage.resize((416,416)) #size=(416,416) or something
    noiseImage = torch.Tensor(noisy(np.array(downImage), 'gauss'))
    #output인 noiseimage는 numpy 형태이므로 다시 tensor로 바꿔주어야합니다.


    '''
    blur종류는 다음과 같습니다. BLUR, Box BLUR, Gaussian BLUR, 블러 정도를 조절하고싶다면 다음을 참고하세요
    documentation : https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html
    noise 종류는 gauss를 하시던지 s&p를 쓰시던지 아무거나 쓰시면 됩니다. 아무래도 색필터가 더 낫지않을까 싶긴합니다
    '''
    return noiseImage