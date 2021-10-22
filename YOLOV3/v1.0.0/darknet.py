from __future__ import division

import numpy as np
import torch
import torch.nn as nn

import utils.dataload as dataload
import utils.layers as layers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Darknet(nn.Module):
    """
    1. DarkNet

    YoloV3의 최종 Network
    """

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.block_list = dataload.cfg_to_block(config_path)
        self.hyperparams, self.module_list = layers.block_to_layer(self.block_list)
        self.yolov3_layers = [layer[0] for layer in self.module_list if isinstance(layer[0], layers.YOLOLayer)]
        self.img_size = img_size

        # 이거는 데이터 로드/세이브할 때 필요하다.
        # 추후에 구현 예정
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x):
        img_size = x.size(2)
        layer_outputs, yolo_outputs = [], []

        for i, (blocks, module) in enumerate(zip(self.block_list[1:], self.module_list)):
            if blocks["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)

            elif blocks["type"] == "route":
                combined_outputs = torch.cat([layer_outputs[int(layer_i)] for layer_i in blocks["layers"].split(",")], 1)
                group_size = combined_outputs.shape[1]
                x = combined_outputs[:, 0: group_size]

            elif blocks["type"] == "shortcut":
                layer_i = int(blocks["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]

            elif blocks["type"] == "yolo":
                x = module[0](x, img_size)
                yolo_outputs.append(x)

            layer_outputs.append(x)

        # Inference와 Training을 구분하여야 한다.
        return yolo_outputs if self.training else torch.cat(yolo_outputs, 1)


        #################
        #   디버깅용!!    #
        #################
        #return yolo_outputs, torch.cat(yolo_outputs, 1)


#################
#   디버깅용!!    #
#################
if __name__ == "__main__":
    from torchsummary import summary

    path2config = "yolov3.cfg"
    model = Darknet(path2config).to(device)
    x = torch.rand(1, 3, 416, 416).to(device)
    with torch.no_grad():
        yolo_outputs, yolo_out_cat = model.forward(x)
        print(yolo_out_cat.shape)
        print(yolo_outputs[0].shape, yolo_outputs[1].shape, yolo_outputs[2].shape)

    summary(model, (3, 416, 416))