"""
DenseBlock 并未按原论文的方式实现，原论文是 做通道 concat，而这里是简化了直接使用 reset方式 做 add
"""
from torch import nn
import torch
from torch.nn import functional as F
from collections import OrderedDict

from .resnet import CBR,Stem,Classify
from .common import Flatten,CBAM

class DenseBlock(nn.Module):
    def __init__(self,in_c,out_c,nums=6):
        super().__init__()
        self.nums = nums
        self.m = nn.ModuleList()
        for i in range(nums):
            self.m.append(nn.Sequential(
                CBR(in_c, out_c, 1, 1),
                nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                CBAM(out_c,3,out_c//4)
            ))

    def forward(self,x):
        out = []
        out.append(x)
        for i in range(self.nums):
            out.append(self.m[i](sum(out) if len(out)==1 else F.relu(sum(out))))
        return F.relu(sum(out))


class DenseBlockDW(nn.Module):
    def __init__(self,in_c,out_c,nums=6):
        super().__init__()
        self.nums = nums
        self.m = nn.ModuleList()
        for i in range(nums):
            self.m.append(nn.Sequential(
                CBR(in_c, out_c, 1, 1),
                CBR(out_c, out_c, 3, 1,groups=out_c),
                CBR(out_c, out_c, 1, 1,act=False),
                CBAM(out_c, 3, out_c // 4)
            ))

    def forward(self,x):
        out = []
        out.append(x)
        for i in range(self.nums):
            out.append(self.m[i](sum(out) if len(out)==1 else F.relu(sum(out))))
        return F.relu(sum(out))


class TransitionLayer(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.add_module("tl",nn.Sequential(
            CBR(in_c,out_c,1),
            nn.AvgPool2d(2,2)
        ))

class DensetNet(nn.Module):
    def __init__(self,in_c=3,num_classes=1000,nums=[6,12,24,16],channels=[64,128,256,512]):
        super().__init__()
        self.feature = nn.Sequential(
            OrderedDict([
                ("stem",Stem(in_c,channels[0],7,2)),
                ("layer1",nn.Sequential(DenseBlock(channels[0],channels[0],nums[0]))),
                ("layer2",nn.Sequential(TransitionLayer(channels[0],channels[1]),DenseBlock(channels[1],channels[1],nums[1]))),
                ("layer3",nn.Sequential(TransitionLayer(channels[1],channels[2]),DenseBlock(channels[2],channels[2],nums[2]))),
                ("layer4",nn.Sequential(TransitionLayer(channels[2],channels[3]),DenseBlock(channels[3],channels[3],nums[3]))),
            ]))

        self.cls = Classify(channels[-1],num_classes)

    def forward(self,x):
        x = self.feature(x)
        x = self.cls(x)

        return x


class DensetNetDW(nn.Module):
    def __init__(self,in_c=3,num_classes=1000,nums=[6,12,24,16],channels=[64,128,256,512]):
        super().__init__()
        self.feature = nn.Sequential(
            OrderedDict([
                ("stem",Stem(in_c,channels[0],7,2)),
                ("layer1",nn.Sequential(DenseBlockDW(channels[0],channels[0],nums[0]))),
                ("layer2",nn.Sequential(TransitionLayer(channels[0],channels[1]),DenseBlockDW(channels[1],channels[1],nums[1]))),
                ("layer3",nn.Sequential(TransitionLayer(channels[1],channels[2]),DenseBlockDW(channels[2],channels[2],nums[2]))),
                ("layer4",nn.Sequential(TransitionLayer(channels[2],channels[3]),DenseBlockDW(channels[3],channels[3],nums[3]))),
            ]))

        self.cls = Classify(channels[-1],num_classes)

    def forward(self,x):
        x = self.feature(x)
        x = self.cls(x)

        return x

nums={
    "densenet21":[2,2,2,2],
    "densenet37":[3,4,6,3],
    "densenet121":[6,12,24,16],
    "densenet169":[6,12,32,32],
    "densenet201":[6,12,48,32],
    "densenet264":[6,12,64,48],
}
channels={
    "ss":[32,48,92,138],
    "s":[32,64,128,256],
    'm':[64,128,256,512],
    'l':[64,256,512,1024]
}

if __name__ == "__main__":
    x = torch.rand([2,3,224,224])
    m = DensetNet(nums=nums['densenet37'],channels=channels['ss'])
    # _initParmas(m.modules())
    print(m(x).shape)
    torch.save(m.state_dict(),'densenet37.pth')