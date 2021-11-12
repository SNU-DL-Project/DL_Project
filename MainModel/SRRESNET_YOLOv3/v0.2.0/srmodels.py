from __future__ import division

import os
import math
import numpy as np

import torch
from torch import nn

from utils.load import layer_block_config
from utils.layers import block_to_layer
from utils.utils import weights_init_normal

class SRRESNET(nn.Module):
    def __init__(self, config_path):
        super(SRRESNET, self).__init__()
        self.block_list = layer_block_config(config_path)
        self.hyperparams, self.module_list = block_to_layer(self.block_list)

        # 이거는 데이터 로드/세이브할 때 필요하다.
        # 추후에 구현 예정
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x):
        layer_outputs = []

        for i, (blocks, module) in enumerate(zip(self.block_list[1:], self.module_list)):
            if blocks["type"] in ["convolutional"]:
                x = module(x)
            elif blocks["type"] == "shortcut":
                layer_i = int(blocks["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            layer_outputs.append(x)

        return (torch.tanh(x)+1)/2

    def load_srres_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        # If the weights file has a cutoff, we can find out about it by looking at the filename
        # examples: darknet53.conv.74 -> cutoff is 74
        filename = os.path.basename(weights_path)
        if ".conv." in filename:
            try:
                cutoff = int(filename.split(".")[-1])  # use last part of filename
            except ValueError:
                pass

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.block_list[1:cutoff], self.module_list[:cutoff])):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b

                    if module_def["activation"] == "P":
                        P_layer = module[2]
                        P_w = torch.from_numpy(weights[ptr:ptr + 1]).view_as(P_layer.weight)
                        P_layer.weight.data.copy_(P_w)
                        ptr += 1
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b

                    if module_def["upsamplescale"]:
                        if module_def["activation"] == "P":
                            P_layer = module[2]
                            P_w = torch.from_numpy(weights[ptr:ptr + 1]).view_as(P_layer.weight)
                            P_layer.weight.data.copy_(P_w)
                            ptr += 1
                    else:
                        if module_def["activation"] == "P":
                            P_layer = module[1]
                            P_w = torch.from_numpy(weights[ptr:ptr + 1]).view_as(P_layer.weight)
                            P_layer.weight.data.copy_(P_w)
                            ptr += 1

                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(
                    weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_srres_weights(self, path, cutoff=None):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        # 순서 : batch/bias -> P -> conv
        for i, (module_def, module) in enumerate(zip(self.block_list[1:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)

                    if module_def["activation"] == "P":
                        P_layer = module[2]
                        P_layer.weight.data.cpu().numpy().tofile(fp)

                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)

                    if module_def["activation"] == "P":
                        if module_def["upsamplescale"]:
                            P_layer = module[2]
                            P_layer.weight.data.cpu().numpy().tofile(fp)
                        else:
                            P_layer = module[1]
                            P_layer.weight.data.cpu().numpy().tofile(fp)

                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

def load_srres_model(model_path_srres, weights_path_srres=None):
    """Loads the yolo model from file.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :return: Returns model
    :rtype: Darknet
    """

    ###문제점 : save load & normal 주기
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select device for inference
    model_srres = SRRESNET(model_path_srres).to(device)

    model_srres.apply(weights_init_normal)
    # If pretrained weights are specified, start from checkpoint or weight file
    if weights_path_srres:
        if weights_path_srres.endswith(".pth"):
            # Load checkpoint weights
            model_srres.load_state_dict(torch.load(weights_path_srres, map_location=device))
        else:
            # Load srganD weights
            model_srres.load_srganD_weights(weights_path_srres)

    return model_srres

'''
class SRGANGenerator(nn.Module):
    def __init__(self, config_path):
        super(SRGANGenerator, self).__init__()
        self.block_list = layer_block_config(config_path)
        self.hyperparams, self.module_list = block_to_layer(self.block_list)
        #self.img_size = 416

        # 이거는 데이터 로드/세이브할 때 필요하다.
        # 추후에 구현 예정
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x):
        layer_outputs = []

        for i, (blocks, module) in enumerate(zip(self.block_list[1:], self.module_list)):
            if blocks["type"] in ["convolutional"]:
                x = module(x)
            elif blocks["type"] == "shortcut":
                layer_i = int(blocks["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            layer_outputs.append(x)

        return (torch.tanh(x)+1)/2

    def load_srganG_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        # If the weights file has a cutoff, we can find out about it by looking at the filename
        # examples: darknet53.conv.74 -> cutoff is 74
        filename = os.path.basename(weights_path)
        if ".conv." in filename:
            try:
                cutoff = int(filename.split(".")[-1])  # use last part of filename
            except ValueError:
                pass

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.block_list[1:cutoff], self.module_list[:cutoff])):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b

                    if module_def["activation"] == "P":
                        P_layer = module[2]
                        P_w = torch.from_numpy(weights[ptr:ptr + 1]).view_as(P_layer.weight)
                        P_layer.weight.data.copy_(P_w)
                        ptr += 1
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b

                    if module_def["upsamplescale"]:
                        if module_def["activation"] == "P":
                            P_layer = module[2]
                            P_w = torch.from_numpy(weights[ptr:ptr + 1]).view_as(P_layer.weight)
                            P_layer.weight.data.copy_(P_w)
                            ptr += 1
                    else:
                        if module_def["activation"] == "P":
                            P_layer = module[1]
                            P_w = torch.from_numpy(weights[ptr:ptr + 1]).view_as(P_layer.weight)
                            P_layer.weight.data.copy_(P_w)
                            ptr += 1

                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(
                    weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_srganG_weights(self, path, cutoff=None):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        # 순서 : batch/bias -> P -> conv
        for i, (module_def, module) in enumerate(zip(self.block_list[1:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)

                    if module_def["activation"] == "P":
                        P_layer = module[2]
                        P_layer.weight.data.cpu().numpy().tofile(fp)

                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)

                    if module_def["activation"] == "P":
                        if module_def["upsamplescale"]:
                            P_layer = module[2]
                            P_layer.weight.data.cpu().numpy().tofile(fp)
                        else:
                            P_layer = module[1]
                            P_layer.weight.data.cpu().numpy().tofile(fp)

                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

class SRGANDiscriminator(nn.Module):
    def __init__(self, config_path):
        super(SRGANDiscriminator, self).__init__()

        self.block_list = layer_block_config(config_path)
        self.hyperparams, self.module_list = block_to_layer(self.block_list)
        # self.img_size = 416

        # 이거는 데이터 로드/세이브할 때 필요하다.
        # 추후에 구현 예정
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x):
        batch_size = x.size(0)
        for i, (blocks, module) in enumerate(zip(self.block_list[1:], self.module_list)):
            if blocks["type"] in ["convolutional", "avgpool"]:
                x = module(x)
        return torch.sigmoid(x.view(batch_size))

    def load_srganD_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        # If the weights file has a cutoff, we can find out about it by looking at the filename
        # examples: darknet53.conv.74 -> cutoff is 74
        filename = os.path.basename(weights_path)
        if ".conv." in filename:
            try:
                cutoff = int(filename.split(".")[-1])  # use last part of filename
            except ValueError:
                pass

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.block_list[1:cutoff], self.module_list[:cutoff])):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(
                    weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_srganD_weights(self, path, cutoff=None):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.block_list[1:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

def load_srgan_model(model_path_D, model_path_G, weights_path_D=None, weights_path_G=None):
    """Loads the yolo model from file.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :return: Returns model
    :rtype: Darknet
    """

    ###문제점 : save load & normal 주기
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select device for inference
    model_D = SRGANDiscriminator(model_path_D).to(device)
    model_G = SRGANGenerator(model_path_G).to(device)

    model_D.apply(weights_init_normal)
    model_G.apply(weights_init_normal)
    # If pretrained weights are specified, start from checkpoint or weight file
    if weights_path_D:
        if weights_path_D.endswith(".pth"):
            # Load checkpoint weights
            model_D.load_state_dict(torch.load(weights_path_D, map_location=device))
        else:
            # Load srganD weights
            model_D.load_srganD_weights(weights_path_D)
    if weights_path_G:
        if weights_path_G.endswith(".pth"):
            # Load checkpoint weights
            model_G.load_state_dict(torch.load(weights_path_G, map_location=device))
        else:
            # Load srganG weights
            model_G.load_srganG_weights(weights_path_G)
    return model_D, model_G
'''


