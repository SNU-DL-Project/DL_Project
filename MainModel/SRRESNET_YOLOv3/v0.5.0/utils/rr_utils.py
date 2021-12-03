import numpy as np
import torch
import torch.nn.functional as F


#2021.12.01 fixed

def _fuse_kernel(kernel, gamma, running_var, eps): #너가 bn을 시키는 놈일까?
    print('fusing: kernel shape', kernel.shape)
    std = np.sqrt(running_var + eps)
    t = gamma / std
    t = np.reshape(t, (-1, 1, 1, 1))
    print('fusing: t', t.shape)
    t = np.tile(t, (1, kernel.shape[1], kernel.shape[2], kernel.shape[3]))
    #t를 커널 사이즈에 맞게 뻥튀기
    return kernel * t

def _fuse_bias(running_mean, running_var, gamma, beta, eps, bias=None):
    if bias is None:
        return beta - running_mean * gamma / np.sqrt(running_var + eps)
    else:
        return beta + (bias - running_mean) * gamma / np.sqrt(running_var + eps)

# fuse conv2d and BN function
def fuse_conv2d_BN(kernel, gamma, running_var, eps, running_mean, beta, bias=None) :
    return _fuse_kernel(kernel, gamma, running_var, eps) + _fuse_bias(running_mean, running_var, gamma, beta, eps, bias=None)


#(layer, depth, w, h ) -> (layer, depth, w, h )
def fuse_conv2d_compactor(kernel, compactor):
    newkernel = F.conv2d(torch.from_numpy(kernel).permute(1, 0, 2, 3), torch.from_numpy(compactor), padding=(0, 0)).permute(1, 0, 2, 3)
    return newkernel

#1. input이 sequential, 사라질 번호 make_compressed_Module
# 바뀌는것 오직 output channel 사이즈
def outchannel_compress(Seq, outchannel_del_list):
    conv=Seq[0]
    oc=conv.out_channels
    conv.out_channels = conv.out_channels-len(outchannel_del_list)
    arr=np.array(range(oc))
    weight=conv.weight

    #compress
    alive_list=np.delete(arr, outchannel_del_list)
    weight=weight[alive_list]

    conv.weight=torch.nn.Parameter(weight)


    return Seq


def inchannel_compress(Seq, inchannel_del_list):
    conv=Seq[0]
    ic=conv.in_channels
    oc=conv.out_channels
    conv.in_channels = ic-len(inchannel_del_list)
    arr=np.array(range(ic))
    weight=conv.weight

    #compress
    if len(arr) < len(inchannel_del_list):
        print('input dimension보다 pruning하려는 개수가 더 큼')
    alive_list=np.delete(arr, inchannel_del_list)

    #자르기 위해 weight분해
    tmp=[]
    for i in range(oc):
        tmp.append(weight[i][alive_list])
    # weight 다시 stack
    weight=torch.stack(tmp)

    #수정된 weight 할당
    conv.weight=torch.nn.Parameter(weight)

    return Seq



