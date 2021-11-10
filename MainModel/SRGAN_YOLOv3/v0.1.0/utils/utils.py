from __future__ import division

import numpy as np
import random

import torch
import torch.nn as nn

def provide_determinism(seed=42):
    """
    1. 고정된 시드를 제공
    고정시드의 경우, 결과를 reproduce 가능
    train.py에서 사용
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def worker_seed_set(worker_id):
    # See for details of numpy:
    # https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    # See for details of random:
    # https://pytorch.org/docs/stable/notes/randomness.html#dataloader

    # NumPy
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))

    # random
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)

def to_cpu(tensor):
    """
    3. cpu로 바꾸기
    """
    return tensor.detach().cpu()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)