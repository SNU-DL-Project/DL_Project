# test.py, main.py는 맨 마지막에 구현

import darknet
import train
import test


config_path = 'dddddd'
data = [] # data는 별도의 모듈에서 진행
YOLO_MODEL = darknet.Darknet(config_path)
train.YOLO_train(YOLO_MODEL, data)