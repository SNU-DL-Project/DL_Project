def layer_block_config(path):
    """
    1. Configuration file to block(model information)

    input : configuration file path
    output : block list

    block : About how to build Network, List of Dictionaries

    block_list[0]에 model hyper parameter 저장
    block_list[1]부터는 layer hyper parameter 저장
    """

    file = open(path, 'r')
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

def data_block_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    with open(path, "r") as fp:
        names = fp.read().splitlines()
    return names