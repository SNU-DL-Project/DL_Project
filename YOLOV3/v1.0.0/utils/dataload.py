from torch.utils.data import DataLoader
from .utils import worker_seed_set

def cfg_to_block(cfgfile):
    """
    1. Configuration file to block(model information)

    input : configuration file (location)
    output : block list

    block : About how to build Network, List of Dictionaries

    block_list[0]에 model hyper parameter 저장
    block_list[1]부터는 layer hyper parameter 저장
    """

    file = open(cfgfile, 'r')
    line_list = file.read().split('\n') # file to list
    line_list = [it for it in line_list if len(it) and not it.startswith('#')]    # 빈 줄과 주석 제거
    line_list = [it.rstrip().lstrip() for it in line_list]  # empty character

    block = {}  # dictionary
    block_list = [] # list (of dictionaries)

    for line in line_list:
        if line.startswith('['):  # New block
            if len(block) != 0:  # block이 비어있지 않다면, 이전 block뒤에 값을 추가한다
                block_list.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    block_list.append(block)

    return block_list

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    with open(path, "r") as fp:
        names = fp.read().splitlines()
    return names

def create_train_data_loader(dataset, batch_size, img_size, n_cpu, multiscale_training=False):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """

    # dataset : data 파일(이미지 파일, target 파일)을 가져온 것
    dataset

    '''
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    '''
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader

def create_validation_data_loader(dataset, batch_size, img_size, n_cpu):
    """
    Creates a DataLoader for validation.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """

    '''
    dataset = ListDataset(img_path, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
    '''
    # dataset : val dataset
    dataset
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader

# 추가적으로 구현해야할 것 : save, load 관련
# augmentation된 데이터를 불러올 때