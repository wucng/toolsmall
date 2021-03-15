from torch import nn
import torch
from torch.nn import functional as F

from .resnet import CBR,Classify
from .common import ChannelShuffle,Flatten,channel_shuffle


class Bottleneck(nn.Module):
    def __init__(self,in_c,hide_c,out_c,stride=1,groups=4):
        super().__init__()
        self.m = nn.Sequential(
            CBR(in_c,hide_c,1,groups=groups),
            ChannelShuffle(groups),
            CBR(hide_c,hide_c,3,stride,groups=hide_c),
            CBR(hide_c, out_c, 1, groups=groups,act=False),
        )
        self.stride = stride

        self.avgpool = nn.AvgPool2d(3,2,1)

    def forward(self,x):
        if self.stride==1:
            return F.relu(self.m(x)+x)
        else:
            return F.relu(torch.cat((self.avgpool(x),self.m(x)),1))

class ShuffleNetv1(nn.Module):
    def __init__(self,in_c=3,num_classes=1000,groups=4):
        super().__init__()
        self.stem = nn.Sequential(
            CBR(in_c,24,3,2),
            nn.MaxPool2d(3,2,1),
        )

        self.stage2 = nn.Sequential(
            Bottleneck(24,272,272-24,2,groups),
            Bottleneck(272, 272, 272, 1, groups),
            Bottleneck(272, 272, 272, 1, groups),
            Bottleneck(272, 272, 272, 1, groups),
        )
        self.stage3 = nn.Sequential(
            Bottleneck(272, 544, 544-272, 2, groups),
            Bottleneck(544, 544, 544, 1, groups),
            Bottleneck(544, 544, 544, 1, groups),
            Bottleneck(544, 544, 544, 1, groups),
            Bottleneck(544, 544, 544, 1, groups),
            Bottleneck(544, 544, 544, 1, groups),
            Bottleneck(544, 544, 544, 1, groups),
            Bottleneck(544, 544, 544, 1, groups),
        )

        self.stage4 = nn.Sequential(
            Bottleneck(544, 1088, 1088-544, 2, groups),
            Bottleneck(1088, 1088, 1088, 1, groups),
            Bottleneck(1088, 1088, 1088, 1, groups),
            Bottleneck(1088, 1088, 1088, 1, groups),
        )

        self.cls = Classify(1088,num_classes)

    def forward(self,x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.cls(x)

        return x

# -------------------shuffleNetv2---------------------------

class BasicBlock(nn.Module):
    def __init__(self,in_c,hide_c,out_c,groups=2):
        super().__init__()
        self.groups = groups
        self.m = nn.Sequential(
            CBR(in_c//groups,hide_c,1),
            CBR(hide_c,hide_c,3,groups=hide_c),
            CBR(hide_c, out_c//groups, 1),
        )
    def forward(self,x):
        x1,x2 = torch.chunk(x,self.groups,1)
        x = torch.cat((x1,self.m(x2)),1)
        return channel_shuffle(x,self.groups)

class DownSampleBlock(nn.Module):
    def __init__(self,in_c,hide_c,out_c,groups=2):
        super().__init__()
        self.groups = groups
        tmp = out_c//2
        self.m1 = nn.Sequential(
            CBR(in_c, in_c, 3,2, groups=in_c),
            CBR(in_c, tmp, 1),
        )
        self.m2 = nn.Sequential(
            CBR(in_c, hide_c, 1),
            CBR(hide_c, hide_c, 3, 2, groups=hide_c),
            CBR(hide_c, tmp, 1),
        )
    def forward(self,x):
        x = torch.cat((self.m1(x), self.m2(x)), 1)
        return channel_shuffle(x, self.groups)


class ShuffleNetv2(nn.Module):
    def __init__(self,in_c=3,num_classes=1000,groups=2):
        super().__init__()
        self.stem = nn.Sequential(
            CBR(in_c,24,3,2),
            nn.MaxPool2d(3,2,1),
        )

        self.stage2 = nn.Sequential(
            DownSampleBlock(24,116,116),
            BasicBlock(116,116,116,groups),
            BasicBlock(116,116,116,groups),
            BasicBlock(116,116,116,groups),
        )
        self.stage3 = nn.Sequential(
            DownSampleBlock(116, 232, 232),
            BasicBlock(232, 232, 232, groups),
            BasicBlock(232, 232, 232, groups),
            BasicBlock(232, 232, 232, groups),
            BasicBlock(232, 232, 232, groups),
            BasicBlock(232, 232, 232, groups),
            BasicBlock(232, 232, 232, groups),
            BasicBlock(232, 232, 232, groups),
        )

        self.stage4 = nn.Sequential(
            DownSampleBlock(232, 464, 464),
            BasicBlock(464, 464, 464, groups),
            BasicBlock(464, 464, 464, groups),
            BasicBlock(464, 464, 464, groups),
            CBR(464,1024,1),
        )

        self.cls = Classify(1024,num_classes)

    def forward(self,x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.cls(x)

        return x

if __name__ == "__main__":
    x = torch.rand([2,3,224,224])
    m = ShuffleNetv2()
    # _initParmas(m.modules())
    print(m(x).shape)
    torch.save(m.state_dict(),'ShuffleNetv2.pth')
