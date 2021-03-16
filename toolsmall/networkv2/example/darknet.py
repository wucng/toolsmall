from .common import Flatten,CBAM
from .yolov5 import SPP,BottleneckCSP,Conv,Focus

from torch import nn
import torch
from torch.nn import functional as F
from collections import OrderedDict

__all__=["CBL","DarknetBlock","DarknetBlockDW","Darknet19","Darknet53","CSPDarknet53","Yolov3SPP","Yolov4"]


class CBL(nn.Module):
    def __init__(self,in_c,out_c,ksize,stride=1,padding=None,dilation=1,groups=1,bias=False,act=True):# 使用BN bias可以设置为False
        super().__init__()
        if padding is None:_ksize = ksize + 2*(dilation-1)
        self.cbl = nn.Sequential(
            nn.Conv2d(in_c,out_c,ksize,stride,padding if padding is not None else _ksize//2,dilation,groups,bias),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1) if act else nn.Identity()
        )
    def forward(self,x):
        return self.cbl(x)

class Classify(nn.Sequential):
    def __init__(self,in_c,num_classes):
        super().__init__()
        self.add_module("cls",nn.Sequential(
            nn.Conv2d(in_c,num_classes,7),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            # nn.Softmax(1)
        ))

class DarknetBlock(nn.Module):
    def __init__(self,in_c,hide_c,out_c):
        super().__init__()
        self.m = nn.Sequential(
            CBL(in_c,hide_c,1),
            CBL(hide_c,out_c,3,act=False),
            CBAM(out_c,3,out_c//4)
        )
    def forward(self,x):
        return F.leaky_relu(self.m(x)+x,0.1)


class DarknetBlockDW(nn.Module):
    def __init__(self,in_c,hide_c,out_c):
        super().__init__()
        self.m = nn.Sequential(
            CBL(in_c,hide_c,1),
            CBL(hide_c,hide_c,3,groups=hide_c), # DWconv
            CBL(hide_c, out_c, 1,act=False),
            CBAM(out_c,3,out_c//4)
        )
    def forward(self,x):
        return F.leaky_relu(self.m(x)+x,0.1)


class Darknet19(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()

        self.stem = nn.Sequential(
            CBL(in_c, 32, 3),
            nn.MaxPool2d(2, 2),
            CBL(32, 64, 3),
            nn.MaxPool2d(2, 2),
        )

        self.layer1 = nn.Sequential(
            CBL(64, 128, 3),
            CBL(128, 64, 1),
            CBL(64, 128, 3),
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            CBL(128, 256, 3),
            CBL(256, 128, 1),
            CBL(128, 256, 3),
        )

        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            CBL(256, 512, 3),
            CBL(512, 256, 1),
            CBL(256, 512, 3),
            CBL(512, 256, 1),
            CBL(256, 512, 3),
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            CBL(512, 1024, 3),
            CBL(1024, 512, 1),
            CBL(512, 1024, 3),
            CBL(1024, 512, 1),
            CBL(512, 1024, 3),
        )

        self.cls = Classify(1024,num_classes)

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cls(x)
        return x


class Darknet19DW(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()

        self.stem = nn.Sequential(
            CBL(in_c, 32, 3),
            nn.MaxPool2d(2, 2),
            CBL(32, 64, 3,groups=32),
            nn.MaxPool2d(2, 2),
        )

        self.layer1 = nn.Sequential(
            CBL(64, 128, 3,groups=32),
            CBL(128, 64, 1),
            CBL(64, 128, 3,groups=32),
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            CBL(128, 256, 3,groups=64),
            CBL(256, 128, 1),
            CBL(128, 256, 3,groups=64),
        )

        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            CBL(256, 512, 3,groups=128),
            CBL(512, 256, 1),
            CBL(256, 512, 3,groups=128),
            CBL(512, 256, 1),
            CBL(256, 512, 3,groups=128),
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            CBL(512, 1024, 3,groups=256),
            CBL(1024, 512, 1),
            CBL(512, 1024, 3,groups=256),
            CBL(1024, 512, 1),
            CBL(512, 1024, 3,groups=256),
        )

        self.cls = Classify(1024,num_classes)

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cls(x)
        return x

class Darknet53(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            CBL(in_c,32,3),
            CBL(32,64,3,2),
            DarknetBlock(64,32,64)
        )
        self.layer1 = nn.Sequential(
            CBL(64,128,3,2),
            DarknetBlock(128,64,128),
            DarknetBlock(128,64,128),
        )

        self.layer2 = nn.Sequential(
            CBL(128,256,3,2),
            DarknetBlock(256,128,256),
            DarknetBlock(256,128,256),
            DarknetBlock(256,128,256),
            DarknetBlock(256,128,256),
            DarknetBlock(256,128,256),
            DarknetBlock(256,128,256),
            DarknetBlock(256,128,256),
            DarknetBlock(256,128,256),
        )

        self.layer3 = nn.Sequential(
            CBL(256,512,3,2),
            DarknetBlock(512,256,512),
            DarknetBlock(512,256,512),
            DarknetBlock(512,256,512),
            DarknetBlock(512,256,512),
            DarknetBlock(512,256,512),
            DarknetBlock(512,256,512),
            DarknetBlock(512,256,512),
            DarknetBlock(512,256,512),
        )

        self.layer4 = nn.Sequential(
            CBL(512,1024,3,2),
            DarknetBlock(1024,512,1024),
            DarknetBlock(1024,512,1024),
            DarknetBlock(1024,512,1024),
            DarknetBlock(1024,512,1024),
        )

        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.Linear(1024,num_classes),
            # nn.Softmax(1)
        )

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cls(x)
        return x


class Darknet37(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            CBL(in_c,32,3),
            CBL(32,64,3,2),
            DarknetBlock(64,32,64)
        )
        self.layer1 = nn.Sequential(
            CBL(64,128,3,2),
            DarknetBlock(128,64,128),
            DarknetBlock(128,64,128),
        )

        self.layer2 = nn.Sequential(
            CBL(128,256,3,2),
            DarknetBlock(256,128,256),
            DarknetBlock(256,128,256),
            DarknetBlock(256,128,256),
            DarknetBlock(256,128,256),
            # DarknetBlock(256,128,256),
            # DarknetBlock(256,128,256),
            # DarknetBlock(256,128,256),
            # DarknetBlock(256,128,256),
        )

        self.layer3 = nn.Sequential(
            CBL(256,512,3,2),
            DarknetBlock(512,256,512),
            DarknetBlock(512,256,512),
            DarknetBlock(512,256,512),
            DarknetBlock(512,256,512),
            DarknetBlock(512,256,512),
            # DarknetBlock(512,256,512),
            # DarknetBlock(512,256,512),
            # DarknetBlock(512,256,512),
        )

        self.layer4 = nn.Sequential(
            CBL(512,1024,3,2),
            DarknetBlock(1024,512,1024),
            DarknetBlock(1024,512,1024),
            DarknetBlock(1024,512,1024),
            # DarknetBlock(1024,512,1024),
        )

        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.Linear(1024,num_classes),
            # nn.Softmax(1)
        )

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cls(x)
        return x


class Darknet31(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            CBL(in_c,32,3),
            CBL(32,64,3,2),
            DarknetBlock(64,32,64)
        )
        self.layer1 = nn.Sequential(
            CBL(64,128,3,2),
            DarknetBlock(128,64,128),
            DarknetBlock(128,64,128),
        )

        self.layer2 = nn.Sequential(
            CBL(128,256,3,2),
            DarknetBlock(256,128,256),
            DarknetBlock(256,128,256),
            DarknetBlock(256,128,256),
            # DarknetBlock(256,128,256),
            # DarknetBlock(256,128,256),
            # DarknetBlock(256,128,256),
            # DarknetBlock(256,128,256),
            # DarknetBlock(256,128,256),
        )

        self.layer3 = nn.Sequential(
            CBL(256,512,3,2),
            DarknetBlock(512,256,512),
            DarknetBlock(512,256,512),
            DarknetBlock(512,256,512),
            # DarknetBlock(512,256,512),
            # DarknetBlock(512,256,512),
            # DarknetBlock(512,256,512),
            # DarknetBlock(512,256,512),
            # DarknetBlock(512,256,512),
        )

        self.layer4 = nn.Sequential(
            CBL(512,1024,3,2),
            DarknetBlock(1024,512,1024),
            DarknetBlock(1024,512,1024),
            DarknetBlock(1024,512,1024),
            # DarknetBlock(1024,512,1024),
        )

        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.Linear(1024,num_classes),
            # nn.Softmax(1)
        )

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cls(x)
        return x


class Darknet53DW(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            CBL(in_c,32,3),
            CBL(32,64,3,2),
            DarknetBlockDW(64,32,64)
        )
        self.layer1 = nn.Sequential(
            CBL(64,128,3,2),
            DarknetBlockDW(128,64,128),
            DarknetBlockDW(128,64,128),
        )

        self.layer2 = nn.Sequential(
            CBL(128,256,3,2),
            DarknetBlockDW(256,128,256),
            DarknetBlockDW(256,128,256),
            DarknetBlockDW(256,128,256),
            DarknetBlockDW(256,128,256),
            DarknetBlockDW(256,128,256),
            DarknetBlockDW(256,128,256),
            DarknetBlockDW(256,128,256),
            DarknetBlockDW(256,128,256),
        )

        self.layer3 = nn.Sequential(
            CBL(256,512,3,2),
            DarknetBlockDW(512,256,512),
            DarknetBlockDW(512,256,512),
            DarknetBlockDW(512,256,512),
            DarknetBlockDW(512,256,512),
            DarknetBlockDW(512,256,512),
            DarknetBlockDW(512,256,512),
            DarknetBlockDW(512,256,512),
            DarknetBlockDW(512,256,512),
        )

        self.layer4 = nn.Sequential(
            CBL(512,1024,3,2),
            DarknetBlockDW(1024,512,1024),
            DarknetBlockDW(1024,512,1024),
            DarknetBlockDW(1024,512,1024),
            DarknetBlockDW(1024,512,1024),
        )

        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.Linear(1024,num_classes),
            # nn.Softmax(1)
        )

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cls(x)
        return x

class Darknet37DW(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            CBL(in_c,32,3),
            CBL(32,64,3,2),
            DarknetBlockDW(64,32,64)
        )
        self.layer1 = nn.Sequential(
            CBL(64,128,3,2),
            DarknetBlockDW(128,64,128),
            DarknetBlockDW(128,64,128),
        )

        self.layer2 = nn.Sequential(
            CBL(128,256,3,2),
            DarknetBlockDW(256,128,256),
            DarknetBlockDW(256,128,256),
            DarknetBlockDW(256,128,256),
            DarknetBlockDW(256,128,256),
            # DarknetBlockDW(256,128,256),
            # DarknetBlockDW(256,128,256),
            # DarknetBlockDW(256,128,256),
            # DarknetBlockDW(256,128,256),
        )

        self.layer3 = nn.Sequential(
            CBL(256,512,3,2),
            DarknetBlockDW(512,256,512),
            DarknetBlockDW(512,256,512),
            DarknetBlockDW(512,256,512),
            DarknetBlockDW(512,256,512),
            DarknetBlockDW(512,256,512),
            # DarknetBlockDW(512,256,512),
            # DarknetBlockDW(512,256,512),
            # DarknetBlockDW(512,256,512),
        )

        self.layer4 = nn.Sequential(
            CBL(512,1024,3,2),
            DarknetBlockDW(1024,512,1024),
            DarknetBlockDW(1024,512,1024),
            DarknetBlockDW(1024,512,1024),
            # DarknetBlockDW(1024,512,1024),
        )

        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.Linear(1024,num_classes),
            # nn.Softmax(1)
        )

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cls(x)
        return x


class Yolov3SPP(nn.Module):
    def __init__(self,num_classes=21,num_anchor=3):
        super().__init__()
        _m = Darknet53()
        self.backbone=nn.Sequential(
            *[_m.stem,_m.layer1,_m.layer2,_m.layer3,_m.layer4]
        )

        self.layer1 = nn.Sequential(
            CBL(1024,512,1),
            CBL(512,1024,3),
            CBL(1024,512,1),
            SPP(512,2048),
            CBL(2048,512,1),
            CBL(512,1024,3),
            CBL(1024,512,1),

            CBL(512,1024,3),
            nn.Conv2d(1024,num_anchor*(num_classes+5),1)
        )
        self.layer2 = nn.Sequential(
            CBL(512,256,1),
            nn.Upsample(scale_factor=2),

            CBL(768,256,1),
            CBL(256,512,3),
            CBL(512,256,1),
            CBL(256,512,3),
            CBL(512, 256, 1),

            CBL(256, 512, 3),
            nn.Conv2d(512, num_anchor * (num_classes + 5), 1)
        )

        self.layer3 = nn.Sequential(
            CBL(256, 128, 1),
            nn.Upsample(scale_factor=2),

            CBL(384, 128, 1),
            CBL(128, 256, 3),
            CBL(256, 128, 1),
            CBL(128, 256, 3),
            CBL(256, 128, 1),

            CBL(128, 256, 3),
            nn.Conv2d(256, num_anchor * (num_classes + 5), 1)
        )

    def forward(self,x):
        x = self.backbone[:2](x)
        x3 = self.backbone[2](x)
        x4 = self.backbone[3](x3)
        x5 = self.backbone[4](x4)

        _x5 = self.layer1[:-2](x5)
        x5 = self.layer1[-2:](_x5)

        x4 = torch.cat((self.layer2[:2](_x5),x4),1)
        _x4 = self.layer2[2:-2](x4)
        x4 = self.layer2[-2:](_x4)

        x3 = torch.cat((self.layer3[:2](_x4),x3),1)
        x3 = self.layer3[2:](x3)

        return x3,x4,x5


class CSPDarknet53(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            Conv(in_c,32,3),
            Conv(32,64,3,2),
            BottleneckCSP(64,64,1),
        )
        self.layer1 = nn.Sequential(
            Conv(64,128,3,2),
            BottleneckCSP(128, 128, 2),
        )

        self.layer2 = nn.Sequential(
            Conv(128, 256, 3, 2),
            BottleneckCSP(256, 256, 8),
        )

        self.layer3 = nn.Sequential(
            Conv(256, 512, 3, 2),
            BottleneckCSP(512, 512, 8),
        )

        self.layer4 = nn.Sequential(
            Conv(512, 1024, 3, 2),
            BottleneckCSP(1024, 1024, 4),
        )

        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.Linear(1024,num_classes),
            # nn.Softmax(1)
        )

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cls(x)
        return x


class Yolov4(nn.Module):
    def __init__(self,num_classes=21,num_anchor=3):
        super().__init__()
        _m = CSPDarknet53()
        self.backbone=nn.Sequential(
            *[_m.stem,_m.layer1,_m.layer2,_m.layer3,_m.layer4]
        )

        self.layer1 = nn.Sequential(
            CBL(1024,512,1),
            CBL(512,1024,3),
            CBL(1024,512,1),
            SPP(512,2048),
            CBL(2048,512,1),
            CBL(512,1024,3),
            CBL(1024,512,1),
        )

        self.layer11 = nn.Sequential(
            CBL(512, 256, 1),
            nn.Upsample(scale_factor=2),

            CBL(512, 512, 1),

            CBL(768, 256, 1),
            CBL(256, 512, 3),
            CBL(512, 256, 1),
            CBL(256, 512, 3),
            CBL(512, 256, 1),
        )

        self.layer12 = nn.Sequential(
            CBL(256, 128, 1),
            nn.Upsample(scale_factor=2),

            CBL(256, 256, 1),

            CBL(384, 128, 1),
            CBL(128, 256, 3),
            CBL(256, 128, 1),
            CBL(128, 256, 3),
            CBL(256, 128, 1),
        )

        self.layer13 = nn.Sequential(
            CBL(128, 256, 3),
            nn.Conv2d(256, num_anchor * (num_classes + 5), 1)
        )

        self.layer2 = nn.Sequential(
            CBL(128, 128, 3,2,1),

            CBL(384, 128, 1),
            CBL(128, 256, 3),
            CBL(256, 128, 1),
            CBL(128, 256, 3),
            CBL(256, 128, 1),

            CBL(128, 256, 3),
            nn.Conv2d(256, num_anchor * (num_classes + 5), 1)
        )

        self.layer3 = nn.Sequential(
            CBL(128, 256, 3,2,1),

            CBL(768,256,1),
            CBL(256,512,3),
            CBL(512,256,1),
            CBL(256,512,3),
            CBL(512, 256, 1),

            CBL(256, 512, 3),
            nn.Conv2d(512, num_anchor * (num_classes + 5), 1)
        )


    def forward(self,x):
        x = self.backbone[:2](x)
        x3 = self.backbone[2](x)
        x4 = self.backbone[3](x3)
        x5 = self.backbone[4](x4)

        _x5 = self.layer1(x5)

        x4 = torch.cat((self.layer11[:2](_x5),self.layer11[2](x4)),1)
        _x4 = self.layer11[3:](x4)

        x3 = torch.cat((self.layer12[:2](_x4), self.layer12[2](x3)), 1)
        _x3 = self.layer12[3:](x3)

        x3 = self.layer13(_x3)

        x4 = torch.cat((self.layer2[0](_x3),_x4),1)
        _x4 = self.layer2[1:-2](x4)
        x4 = self.layer2[-2:](_x4)

        x5 = torch.cat((self.layer3[0](_x4), _x5), 1)
        x5 = self.layer3[1:](x5)

        return x3,x4,x5


class Yolov4One(nn.Module):
    def __init__(self,num_classes=21,num_anchor=3):
        super().__init__()
        # _m = CSPDarknet53()
        # _m = Darknet37DW()
        _m = Darknet19DW()
        self.backbone=nn.Sequential(
            *[_m.stem,_m.layer1,_m.layer2,_m.layer3,_m.layer4]
        )

        self.layer1 = nn.Sequential(
            CBL(1024,512,1),
            CBL(512,1024,3),
            CBL(1024,512,1),
            SPP(512,2048),
            CBL(2048,512,1),
            CBL(512,1024,3),
            CBL(1024,512,1),
        )

        self.layer11 = nn.Sequential(
            CBL(512, 256, 1),
            nn.Upsample(scale_factor=2),

            CBL(512, 512, 1),

            CBL(768, 256, 1),
            CBL(256, 512, 3),
            CBL(512, 256, 1),
            CBL(256, 512, 3),
            CBL(512, 256, 1),
        )

        self.layer12 = nn.Sequential(
            CBL(256, 128, 1),
            nn.Upsample(scale_factor=2),

            CBL(256, 256, 1),

            CBL(384, 128, 1),
            CBL(128, 256, 3),
            CBL(256, 128, 1),
            CBL(128, 256, 3),
            CBL(256, 128, 1),
        )

        self.layer13 = nn.Sequential(
            CBL(128, 256, 3),
            nn.Conv2d(256, num_anchor * (num_classes + 5), 1)
        )

        # self.layer2 = nn.Sequential(
        #     CBL(128, 128, 3,2,1),
        #
        #     CBL(384, 128, 1),
        #     CBL(128, 256, 3),
        #     CBL(256, 128, 1),
        #     CBL(128, 256, 3),
        #     CBL(256, 128, 1),
        #
        #     CBL(128, 256, 3),
        #     nn.Conv2d(256, num_anchor * (num_classes + 5), 1)
        # )
        #
        # self.layer3 = nn.Sequential(
        #     CBL(128, 256, 3,2,1),
        #
        #     CBL(768,256,1),
        #     CBL(256,512,3),
        #     CBL(512,256,1),
        #     CBL(256,512,3),
        #     CBL(512, 256, 1),
        #
        #     CBL(256, 512, 3),
        #     nn.Conv2d(512, num_anchor * (num_classes + 5), 1)
        # )


    def forward(self,x):
        x = self.backbone[:2](x)
        x3 = self.backbone[2](x)
        x4 = self.backbone[3](x3)
        x5 = self.backbone[4](x4)

        _x5 = self.layer1(x5)

        x4 = torch.cat((self.layer11[:2](_x5),self.layer11[2](x4)),1)
        _x4 = self.layer11[3:](x4)

        x3 = torch.cat((self.layer12[:2](_x4), self.layer12[2](x3)), 1)
        _x3 = self.layer12[3:](x3)

        x3 = self.layer13(_x3)

        # x4 = torch.cat((self.layer2[0](_x3),_x4),1)
        # _x4 = self.layer2[1:-2](x4)
        # x4 = self.layer2[-2:](_x4)
        #
        # x5 = torch.cat((self.layer3[0](_x4), _x5), 1)
        # x5 = self.layer3[1:](x5)

        return x3#,x4,x5

if __name__ == "__main__":
    x = torch.rand([2,3,224,224])
    m = Yolov4()
    # _initParmas(m.modules())
    print(m(x)[-1].shape)
    torch.save(m.state_dict(),'Yolov4.pth')