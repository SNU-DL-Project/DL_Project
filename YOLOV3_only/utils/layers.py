from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class EmptyLayer(nn.Module):
    """
    1. Empty layer
    """
    def __init__(self):
        super().__init__()

class YOLOLayer(nn.Module):
    """
    2. YOLO Layer
    """

    def __init__(self, anchors_list, num_classes, img_dim=416):
        super().__init__()
        """
        anchors_list : anchor들의 list, 각 yolo layer마다 3개의 anchor box가 할당된다.
        num_classes : 구별할 class 개수
        img_dim : YOLO layer에 입력되는 이미지의 가로, 세로 길이
        """

        self.anchors_list = anchors_list
        self.num_anchors = len(anchors_list) # YOLOV3 기반 모델은 3으로 고정된다.
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.img_dim = img_dim
        self.num_outputs = num_classes + 5  # output 개수 (class 수 + xywh(4) + object확률(1))
        self.grid = torch.zeros(1)
        self.grid_size = 0

        anchors = torch.tensor(list(chain(*anchors_list))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None

    def forward(self, x, img_size):
        # x의 shape : (batch_size, channels, H, W)
        batch_size, _, ny, nx = x.shape
        stride = img_size // x.size(2)
        self.stride = stride
        x = x.view(batch_size, self.num_anchors, self.num_outputs, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        # inference가 들어온 경우, 출력에 맞게 x값들을 변환해주어야 한다.
        if not self.training:
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)

            x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # xy
            x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid # wh
            x[..., 4:] = x[..., 4:].sigmoid()
            x = x.view(batch_size, -1, self.num_outputs)

        return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def block_to_layer(module_defs):
    """
    3. block to layer

    input : block list
    output : module list
    """
    hyperparams = module_defs[0]
    hyperparams.update({
        'batch': int(hyperparams['batch']),
        'subdivisions': int(hyperparams['subdivisions']),
        'width': int(hyperparams['width']),
        'height': int(hyperparams['height']),
        'channels': int(hyperparams['channels']),
        'optimizer': hyperparams.get('optimizer'),
        'momentum': float(hyperparams['momentum']),
        'decay': float(hyperparams['decay']),
        'learning_rate': float(hyperparams['learning_rate']),
        'burn_in': int(hyperparams['burn_in']),
        'max_batches': int(hyperparams['max_batches']),
        'policy': hyperparams['policy'],
        'lr_steps': list(zip(map(int,   hyperparams["steps"].split(",")),
                             map(float, hyperparams["scales"].split(","))))
    })
    assert hyperparams["height"] == hyperparams["width"], \
        "Height and width should be equal! Non square images are padded with zeros."
    output_filters = [hyperparams["channels"]]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs[1:]):
        modules = nn.Sequential()
        if module_def["type"] == "convolutional":
            try:
                bn = int(module_def["batch_normalize"])
                bias = False
            except:
                bn = 0
                bias = True
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=bias,
                ),
            )

            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.1, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
            if module_def["activation"] == "leaky2":
                modules.add_module(f"leaky2_{module_i}", nn.LeakyReLU(0.2))
            if module_def["activation"] == "P":
                modules.add_module(f"P_{module_i}", nn.PReLU())

            #activation : linear, vartanh, sigmoid는 안건드림
            #if module_def["activation"] == "mish":
            #    modules.add_module(f"mish_{module_i}", Mish())

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                                   padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        #SRGAN
        elif module_def["type"] == "avgpool":
            avgpool = nn.AdaptiveAvgPool2d(int(module_def["output_size"]))
            modules.add_module(f"avgpool_{module_i}", avgpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        #SRGAN
        elif module_def["type"] == "upsampleblock":
            upsampleblock = UpsampleBLock(int(module_def["filters"]), int(module_def["scale"]))
            modules.add_module(f"upsampleblock_{module_i}", upsampleblock)

        #SRGAN
        elif module_def["type"] == "residualblock":
            residualblock = ResidualBlock(int(module_def["filters"]))
            modules.add_module(f"residualblock_{module_i}", residualblock)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers]) // int(module_def.get("groups", 1))
            modules.add_module(f"route_{module_i}", nn.Sequential())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", nn.Sequential())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list