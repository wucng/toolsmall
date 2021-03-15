import torch
from torch import nn
from torch.nn import functional as F

from .common import Flatten
from .resnet import Classify,Downsample

class CBR(nn.Module):
    def __init__(self,in_c,out_c,ksize,stride=1,padding=None,dilation=1,groups=1,bias=False,act=True):# 使用BN bias可以设置为False
        super().__init__()
        if padding is None:_ksize = ksize + 2*(dilation-1)
        self.cbr = nn.Sequential(
            nn.Conv2d(in_c,out_c,ksize,stride,padding if padding is not None else _ksize//2,dilation,groups,bias),
            nn.BatchNorm2d(out_c),
            nn.ReLU6() if act else nn.Identity()
        )
    def forward(self,x):
        return self.cbr(x)

class MobileNetV1(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.feature = nn.Sequential(
            CBR(in_c,32,3,2),
            CBR(32,32,3,groups=32),
            CBR(32, 64, 1),
            CBR(64, 64, 3,2, groups=64),
            CBR(64, 128, 1),
            CBR(128, 128, 3, groups=128),
            CBR(128, 128, 1),

            CBR(128, 128, 3, 2, groups=128),
            CBR(128, 256, 1),
            CBR(256, 256, 3, groups=256),
            CBR(256, 256, 1),

            CBR(256, 256, 3, 2, groups=256),
            CBR(256, 512, 1),
            CBR(512, 512, 3, groups=512),
            CBR(512, 512, 1),
            CBR(512, 512, 3, groups=512),
            CBR(512, 512, 1),
            CBR(512, 512, 3, groups=512),
            CBR(512, 512, 1),
            CBR(512, 512, 3, groups=512),
            CBR(512, 512, 1),
            CBR(512, 512, 3, groups=512),
            CBR(512, 512, 1),

            CBR(512, 512, 3, 2, groups=512),
            CBR(512, 1024, 1),
            CBR(1024, 1024, 3, 1, groups=1024),
            CBR(1024, 1024, 1),
        )

        self.cls = Classify(1024,num_classes)

    def forward(self,x):
        x = self.feature(x)
        x = self.cls(x)

        return x

class Bottleneck(nn.Module):
    def __init__(self,in_c,hide_c,out_c,ksize=3,stride=1):
        super().__init__()
        self.m = nn.Sequential(
            CBR(in_c,hide_c,1),
            CBR(hide_c,hide_c,ksize,stride,groups=hide_c),
            CBR(hide_c,out_c,1,act=False)
        )
        self.stride=stride

    def forward(self,x):
        if self.stride >1:
            return self.m(x)
        else:
            return self.m(x)+x

class Bottleneckv2(nn.Module):
    def __init__(self,in_c,hide_c,out_c,ksize=3,stride=1,useSE=True):
        super().__init__()
        self.m = nn.Sequential(
            CBR(in_c,hide_c,1),
            CBR(hide_c,hide_c,ksize,stride,groups=hide_c),
            SEblock(hide_c,hide_c//4,hide_c) if useSE else nn.Identity(),
            CBR(hide_c,out_c,1,act=False)
        )
        self.downsample = None
        if stride > 1 or in_c!=out_c:
            self.downsample = Downsample(in_c, out_c,1,stride)

    def forward(self, x):
        return F.relu(self.m(x) + x if self.downsample is None else self.downsample(x))

class MobileNetV2(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.feature = nn.Sequential(
            CBR(in_c,32,3,2),
            Bottleneckv2(32,32,16,3,1),

            Bottleneckv2(16,16*6,24,3,2),
            Bottleneckv2(24,24*6,24,3,1),

            Bottleneckv2(24, 24 * 6, 32, 3, 2),
            Bottleneckv2(32, 32 * 6, 32, 3, 1),
            Bottleneckv2(32, 32 * 6, 32, 3, 1),

            Bottleneckv2(32, 32 * 6, 64, 3, 2),
            Bottleneckv2(64, 64 * 6, 64, 3, 1),
            Bottleneckv2(64, 64 * 6, 64, 3, 1),
            Bottleneckv2(64, 64 * 6, 64, 3, 1),

            Bottleneckv2(64, 64 * 6, 96, 3, 1),
            Bottleneckv2(96, 96 * 6, 96, 3, 1),
            Bottleneckv2(96, 96 * 6, 96, 3, 1),

            Bottleneckv2(96, 96 * 6, 160, 3, 2),
            Bottleneckv2(160, 160 * 6, 160, 3, 1),
            Bottleneckv2(160, 160 * 6, 160, 3, 1),

            Bottleneckv2(160, 160 * 6, 320, 3, 1),

            CBR(320, 1280, 1, 1),
        )
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(1280,num_classes,1),
            Flatten()
        )

    def forward(self,x):
        x = self.feature(x)
        x = self.cls(x)
        return x


class CBH(nn.Module):
    def __init__(self,in_c,out_c,ksize,stride=1,padding=None,dilation=1,groups=1,bias=False,act=True):# 使用BN bias可以设置为False
        super().__init__()
        if padding is None:_ksize = ksize + 2*(dilation-1)
        self.cbr = nn.Sequential(
            nn.Conv2d(in_c,out_c,ksize,stride,padding if padding is not None else _ksize//2,dilation,groups,bias),
            nn.BatchNorm2d(out_c),
            nn.Hardswish() if act else nn.Identity()
        )
    def forward(self,x):
        return self.cbr(x)

class SEblock(nn.Module):
    def __init__(self,in_c,hide_c,out_c):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_c,hide_c),
            nn.ReLU(),
            nn.Linear(hide_c,out_c),
            nn.Hardswish()
        )
    def forward(self,x):
        x1 = x.mean([2,3])
        x1 = self.l1(x1)
        return x*x1[...,None,None]


class BottleneckSE(nn.Module):
    def __init__(self,in_c,hide_c,out_c,ksize=3,stride=1):
        super().__init__()
        self.m = nn.Sequential(
            CBR(in_c,hide_c,1),
            CBR(hide_c,hide_c,ksize,stride,groups=hide_c),
            # SE block
            SEblock(hide_c,hide_c//4,hide_c),
            CBR(hide_c,out_c,1,act=False)
        )
        self.downsample = None
        if stride > 1 or in_c!=out_c:
            self.downsample = Downsample(in_c, out_c,1,stride)

    def forward(self, x):
        return F.relu(self.m(x) + x if self.downsample is None else self.downsample(x))

class MobileNetV3(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.feature = nn.Sequential(
            CBH(in_c,16,3,2),
            BottleneckSE(16,16,16,3,2),
            BottleneckSE(16,72,24,3,2),
            BottleneckSE(24,88,24,3,1),
            BottleneckSE(24,96,40,3,2),
            BottleneckSE(40,240,40,3,1),
            BottleneckSE(40,240,40,3,1),
            BottleneckSE(40,120,48,3,1),
            BottleneckSE(48,144,48,3,1),
            BottleneckSE(48,288,96,3,2),
            BottleneckSE(96,576,96,3,1),
            BottleneckSE(96,576,96,3,1),
            CBH(96,576,1),

        )
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(576, 1024, 1),
            nn.Hardswish(),
            nn.Conv2d(1024, num_classes, 1),
            Flatten()
        )

    def forward(self,x):
        x = self.feature(x)
        x = self.cls(x)
        return x


if __name__ == "__main__":
    x = torch.rand([2,3,224,224])
    m = MobileNetV2()
    # _initParmas(m.modules())
    print(m(x).shape)
    torch.save(m.state_dict(),'MobileNetV2.pth')