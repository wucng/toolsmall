from .common import Flatten,CBAM

from torch import nn
import torch
from torch.nn import functional as F

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
        self.feature = nn.Sequential(
            CBL(in_c,32,3),
            nn.MaxPool2d(2,2),
            CBL(32, 64, 3),
            nn.MaxPool2d(2, 2),
            CBL(64, 128, 3),
            CBL(128, 64, 1),
            CBL(64, 128, 3),

            nn.MaxPool2d(2, 2),
            CBL(128, 256, 3),
            CBL(256, 128, 1),
            CBL(128, 256, 3),

            nn.MaxPool2d(2, 2),
            CBL(256, 512, 3),
            CBL(512, 256, 1),
            CBL(256, 512, 3),
            CBL(512, 256, 1),
            CBL(256, 512, 3),

            nn.MaxPool2d(2, 2),
            CBL(512, 1024, 3),
            CBL(1024, 512, 1),
            CBL(512, 1024, 3),
            CBL(1024, 512, 1),
            CBL(512, 1024, 3),
        )

        self.cls = Classify(1024,num_classes)

    def forward(self,x):
        x = self.feature(x)
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

if __name__ == "__main__":
    x = torch.rand([2,3,224,224])
    m = Darknet53()
    # _initParmas(m.modules())
    print(m(x).shape)
    torch.save(m.state_dict(),'Darknet53.pth')