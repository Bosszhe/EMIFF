from mmcv.cnn import ConvModule
from torch import nn
from mmcv.ops import DeformConv2dPack as DCN

from ..builder import NECKS

BN_MOMENTUM = 0.1

@NECKS.register_module()
class DeformConv(nn.Module):
    def __init__(self, 
                    chi, 
                    cho,
                    kernel_size=(3,3),
                    stride=1,
                    padding=1,
                    dilation=1,
                    deform_groups=1):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, deform_groups=deform_groups)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x