# test.py, main.py는 맨 마지막에 구현

import utils.load
import darknet
import train
import torch
# import test

if __name__ == "__main__":

    data_path = 'config/custom.txt'
    model_path = 'config/yolov3.cfg'
    data_config = utils.load.data_block_config(data_path)
    YOLO_MODEL = darknet.load_yolo_model(model_path)
    TEST_YOLO_TRAIN = train.YOLO_train(YOLO_MODEL, data_config)
    TEST_YOLO_TRAIN.run(epochs=200)

    '''
    data_path = 'config/custom.txt'
    model_path = 'config/yolov3.cfg'
    data_config = utils.load.data_block_config(data_path)
    YOLO_MODEL = darknet.load_yolo_model(model_path)
    TEST_YOLO_TRAIN = train.YOLO_train(YOLO_MODEL, data_config)
    TEST_YOLO_TRAIN.run(epochs=200)
    '''
