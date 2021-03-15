# from torchvision.models.resnet import BasicBlock,Bottleneck
# from torchvision.models.squeezenet import Fire
# from torchvision.models.densenet import _DenseBlock
# from torchvision.models.mobilenet import InvertedResidual,ConvBNReLU
# # from torchvision.models.shufflenetv2 import channel_shuffle,InvertedResidual
from .common import Flatten

from torch import nn
import torch
from torch.nn import functional as F

class CBR(nn.Module):
    def __init__(self,in_c,out_c,ksize,stride=1,padding=None,dilation=1,groups=1,bias=False,act=True):# 使用BN bias可以设置为False
        super().__init__()
        if padding is None:_ksize = ksize + 2*(dilation-1)
        self.cbr = nn.Sequential(
            nn.Conv2d(in_c,out_c,ksize,stride,padding if padding is not None else _ksize//2,dilation,groups,bias),
            nn.BatchNorm2d(out_c),
            nn.ReLU() if act else nn.Identity()
        )
    def forward(self,x):
        return self.cbr(x)

class Stem(nn.Sequential):
    def __init__(self,in_c,out_c,ksize=7,stride=2):
        super().__init__()
        self.add_module('stem',nn.Sequential(CBR(in_c,out_c,ksize,stride,ksize//2),nn.MaxPool2d(3,2,3//2)))

class Downsample(nn.Sequential):
    def __init__(self,in_c,out_c,ksize=1,stride=2):
        super().__init__()
        self.add_module('downsample', nn.Sequential(nn.Conv2d(in_c,out_c,ksize,stride,bias=False),
                nn.BatchNorm2d(out_c)))


class BasicBlock(nn.Module):
    def __init__(self,in_c,hide_c,out_c,ksize=3,stride=1):
        super().__init__()
        self.m = nn.Sequential(
            CBR(in_c,hide_c,ksize,stride,padding=ksize//2),
            nn.Conv2d(hide_c, out_c, ksize, stride=1, padding=ksize//2, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.downsample = None
        if stride > 1 or in_c!=out_c:
            self.downsample = Downsample(in_c,out_c,1,stride)

    def forward(self,x):
        return F.relu(self.m(x) + x if self.downsample is None else self.downsample(x))


class Bottleneck(nn.Module):
    def __init__(self,in_c,hide_c,out_c,ksize=3,stride=1):
        super().__init__()
        self.m = nn.Sequential(
            CBR(in_c, hide_c, 1, 1, bias=False),
            CBR(hide_c, hide_c, ksize, stride, padding=ksize // 2),
            nn.Conv2d(hide_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.downsample = None
        if stride > 1 or in_c!=out_c:
            self.downsample = Downsample(in_c, out_c,1,stride)

    def forward(self, x):
        return F.relu(self.m(x) + x if self.downsample is None else self.downsample(x))

class Classify(nn.Sequential):
    def __init__(self,in_c,num_classes):
        super().__init__()
        self.add_module("cls",nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_c, num_classes)
        ))

        # self.add_module("cls", nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(in_c,num_classes,1), # 1x1卷积替换 fc层
        #     Flatten()
        # ))

        # self.add_module("cls", nn.Sequential(
        #     nn.Conv2d(in_c,num_classes,1), # 1x1卷积替换 fc层
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     Flatten()
        # ))

class Resnet22(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = Stem(in_c,64,7,2)
        self.layer1 = nn.Sequential(
            BasicBlock(64,64,64,3,1),
            Bottleneck(64, 16, 64, 3, 1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64,128,128,3,2),
            Bottleneck(128, 32, 128, 3, 1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, 256, 3, 2),
            Bottleneck(256, 64, 256, 3, 1),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, 512, 3, 2),
            Bottleneck(512, 128, 512, 3, 1),
        )

        self.cls = Classify(512,num_classes)

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cls(x)

        return x

class Resnet12(nn.Module): # 效果很差
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = Stem(in_c,64,7,2)
        self.layer1 = nn.Sequential(
            BasicBlock(64,64,64,3,1),
        )
        self.layer2 = nn.Sequential(
            Bottleneck(64, 32, 128, 3, 2),
        )
        self.layer3 = nn.Sequential(
            Bottleneck(128, 64, 256, 3, 2),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, 512, 3, 2),
        )

        self.cls = Classify(512,num_classes)

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cls(x)

        return x

class Resnet18(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = Stem(in_c,64,7,2)
        self.layer1 = nn.Sequential(
            BasicBlock(64,64,64,3,1),
            BasicBlock(64,64,64,3,1)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64,128,128,3,2),
            BasicBlock(128,128,128,3,1)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, 256, 3, 2),
            BasicBlock(256, 256, 256, 3, 1)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, 512, 3, 2),
            BasicBlock(512, 512, 512, 3, 1)
        )

        self.cls = Classify(512,num_classes)

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cls(x)

        return x

class Resnet34(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = Stem(in_c,64,7,2)
        self.layer1 = nn.Sequential(
            BasicBlock(64,64,64,3,1),
            BasicBlock(64,64,64,3,1),
            BasicBlock(64,64,64,3,1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64,128,128,3,2),
            BasicBlock(128,128,128,3,1),
            BasicBlock(128,128,128,3,1),
            BasicBlock(128,128,128,3,1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, 256, 3, 2),
            BasicBlock(256, 256, 256, 3, 1),
            BasicBlock(256, 256, 256, 3, 1),
            BasicBlock(256, 256, 256, 3, 1),
            BasicBlock(256, 256, 256, 3, 1),
            BasicBlock(256, 256, 256, 3, 1),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, 512, 3, 2),
            BasicBlock(512, 512, 512, 3, 1),
            BasicBlock(512, 512, 512, 3, 1),
        )

        self.cls = Classify(512,num_classes)

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cls(x)

        return x

class Resnet50(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = Stem(in_c,64,7,2)
        self.layer1 = nn.Sequential(
            Bottleneck(64,64,256,3,1),
            Bottleneck(256,64,256,3,1),
            Bottleneck(256,64,256,3,1),
        )
        self.layer2 = nn.Sequential(
            Bottleneck(256, 128, 512, 3, 2),
            Bottleneck(512, 128, 512, 3, 1),
            Bottleneck(512, 128, 512, 3, 1),
            Bottleneck(512, 128, 512, 3, 1),
        )
        self.layer3 = nn.Sequential(
            Bottleneck(512, 256, 1024, 3, 2),
            Bottleneck(1024, 256, 1024, 3, 1),
            Bottleneck(1024, 256, 1024, 3, 1),
            Bottleneck(1024, 256, 1024, 3, 1),
            Bottleneck(1024, 256, 1024, 3, 1),
            Bottleneck(1024, 256, 1024, 3, 1),
        )
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 512, 2048, 3, 2),
            Bottleneck(2048, 512, 2048, 3, 1),
            Bottleneck(2048, 512, 2048, 3, 1)
        )

        self.cls = Classify(2048,num_classes)

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cls(x)

        return x

class Resnet101(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = Stem(in_c,64,7,2)
        self.layer1 = nn.Sequential(
            Bottleneck(64,64,256,3,1),
            Bottleneck(256,64,256,3,1),
            Bottleneck(256,64,256,3,1),
        )
        self.layer2 = nn.Sequential(
            Bottleneck(256, 128, 512, 3, 2),
            Bottleneck(512, 128, 512, 3, 1),
            Bottleneck(512, 128, 512, 3, 1),
            Bottleneck(512, 128, 512, 3, 1),
        )

        tmp = []
        tmp.append(Bottleneck(512, 256, 1024, 3, 2))
        for _ in range(22):
            tmp.append(Bottleneck(1024, 256, 1024, 3, 1))
        self.layer3 = nn.Sequential(*tmp)

        self.layer4 = nn.Sequential(
            Bottleneck(1024, 512, 2048, 3, 2),
            Bottleneck(2048, 512, 2048, 3, 1),
            Bottleneck(2048, 512, 2048, 3, 1)
        )

        self.cls = Classify(2048,num_classes)

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cls(x)

        return x

class Resnet152(nn.Module):
    def __init__(self, in_c=3, num_classes=1000):
        super().__init__()
        self.stem = Stem(in_c, 64, 7, 2)
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, 256, 3, 1),
            Bottleneck(256, 64, 256, 3, 1),
            Bottleneck(256, 64, 256, 3, 1),
        )

        tmp = []
        tmp.append(Bottleneck(256, 128, 512, 3, 2))
        for _ in range(7):
            tmp.append(Bottleneck(512, 128, 512, 3, 1))
        self.layer2 = nn.Sequential(*tmp)

        tmp = []
        tmp.append(Bottleneck(512, 256, 1024, 3, 2))
        for _ in range(35):
            tmp.append(Bottleneck(1024, 256, 1024, 3, 1))
        self.layer3 = nn.Sequential(*tmp)

        self.layer4 = nn.Sequential(
            Bottleneck(1024, 512, 2048, 3, 2),
            Bottleneck(2048, 512, 2048, 3, 1),
            Bottleneck(2048, 512, 2048, 3, 1)
        )

        self.cls = Classify(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cls(x)

        return x

if __name__ == "__main__":
    x = torch.rand([2,3,224,224])
    # m = Resnet18()
    # _initParmas(m.modules())
    # print(m(x).shape)
    # torch.save(m.state_dict(),'Resnet18.pth')

    m=CBR(3, 32, 3,dilation=2)
    print(m(x).shape)