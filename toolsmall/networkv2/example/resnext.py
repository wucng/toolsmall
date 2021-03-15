from torch import nn
import torch
from torch.nn import functional as F

from .resnet import Stem,CBR,Classify,Downsample
from .common import Flatten


class BasicBlock(nn.Module):
    def __init__(self,in_c,hide_c,out_c,ksize=3,stride=1,groups=32):
        super().__init__()
        self.groups = groups
        hide_c = in_c//groups
        tmp = [nn.Sequential(
            CBR(in_c,hide_c,ksize,stride,padding=ksize//2),
            nn.Conv2d(hide_c, out_c, ksize, stride=1, padding=ksize//2, bias=False),
            nn.BatchNorm2d(out_c),
        ) for _ in range(groups)]

        self.m = nn.Sequential(*tmp)

        self.downsample = None
        if stride > 1 or in_c!=out_c:
            self.downsample = Downsample(in_c,out_c,1,stride)

    def forward(self,x):
        tmp = []
        for i in range(self.groups):
            tmp.append(self.m[i](x))

        return F.relu(F.relu(sum(tmp)) + x if self.downsample is None else self.downsample(x))


class Bottleneck(nn.Module):
    def __init__(self,in_c,hide_c,out_c,ksize=3,stride=1,groups=32):
        super().__init__()
        self.groups = groups
        self.l1 = CBR(in_c, hide_c, 1)
        self.l2 = CBR(hide_c,in_c, 1,act=False)

        tmp = [CBR(hide_c, hide_c//groups, ksize, stride) for _ in range(groups)]
        self.m = nn.Sequential(*tmp)

        self.downsample = None
        if stride > 1 or in_c!=out_c:
            self.downsample = Downsample(in_c, out_c,1,stride)

    def forward(self, x):
        x1 = self.l1(x)
        tmp = []
        for i in range(self.groups):
            tmp.append(self.m[i](x1))
        x2 = torch.cat(tmp,1)
        x3 = self.l2(x2)

        return F.relu(x3 + x if self.downsample is None else self.downsample(x))


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

if __name__ == "__main__":
    x = torch.rand([2,3,224,224])
    m = Resnet18()
    # _initParmas(m.modules())
    print(m(x).shape)
    torch.save(m.state_dict(),'Resnet18.pth')