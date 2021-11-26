# test.py, main.py는 맨 마지막에 구현

import utils.load
import models
import train
import torch
# import test

if __name__ == "__main__":
    # epochs 횟수 수정 가능
    # 저장 간격 바꾸고 싶을 때 : TEST_YOLO_TRAIN.save_interval = 10 (10회에 1번 저장)
    # eval 간격 바꾸고 싶을 때 : TEST_YOLO_TRAIN.evaluation_interval = 10 (10회에 1번 eval)
    # verbose 끄고 싶을 때 : TEST_YOLO_TRAIN.verbose = 0

    '''
    # YOLOv3
    data_path = 'config/custom.txt'
    model_path = 'config/yolov3.cfg'
    data_config = utils.load.data_block_config(data_path)
    YOLO_MODEL = models.load_yolo_model(model_path)
    YOLO_TRAIN = train.YOLO_train(YOLO_MODEL, data_config)
    YOLO_TRAIN.save_interval = 10
    YOLO_TRAIN.run(epochs=200)
    YOLO_TRAIN.yolo_save('yolo_final.weight')
    '''
    '''
    # SRRESnet
    data_path = 'config/custom.txt'
    model_srres_path = 'config/srres.cfg'
    data_config = utils.load.data_block_config(data_path)
    MODEL_SRRES = models.load_srres_model(model_srres_path)
    SRRES_TRAIN = train.SRRES_train(MODEL_SRRES, data_config)
    SRRES_TRAIN.run(epochs=100)
    SRRES_TRAIN.srres_save('srres_final.weight')
    '''

    # SRRES_YOLO
    data_path = 'config/custom.txt'
    model_srres_path = 'config/srres.cfg'
    model_yolo_path = 'config/yolov3.cfg'
    data_config = utils.load.data_block_config(data_path)
    MODEL_SRRES = models.load_srres_model(model_srres_path)
    MODEL_YOLO = models.load_yolo_model(model_yolo_path)
    SRRES_YOLO_TRAIN = train.SRRES_YOLO_train(MODEL_SRRES, MODEL_YOLO, data_config)
    SRRES_YOLO_TRAIN.run(epochs=100)
    SRRES_YOLO_TRAIN.srres_yolo_save('srres_final.weight', 'yolo_final.weight')


