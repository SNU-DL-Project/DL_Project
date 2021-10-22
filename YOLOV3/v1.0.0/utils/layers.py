import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.stride = None

    def forward(self, x, img_size):
        # x의 shape : (batch_size, channels, H, W)
        batch_size, _, ny, nx = x.shape
        stride = img_size // x.size(2)

        x = x.view(batch_size, self.num_anchors, self.num_outputs, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        # inference가 들어온 경우, 출력에 맞게 x값들을 변환해주어야 한다.
        if not self.training:
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)
            x = x.view(batch_size, self.num_anchors, self.num_outputs, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # xy
            x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid # wh
            x[..., 4:] = x[..., 4:].sigmoid()
            x = x.view(batch_size, -1, self.no)

        return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
