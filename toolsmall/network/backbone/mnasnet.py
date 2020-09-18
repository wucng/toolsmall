from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.nn as nn

from .moudle import SeBlock,DwConv2dV1,_initialize_weights,MixConv2d

class DwConvBNRelu(nn.Module):
    def __init__(self,in_planes, out_planes, kernel_size=3, stride=1,padding=0):
        super().__init__()
        self.m = nn.Sequential(
            # DwConv2dV1(in_planes,out_planes,kernel_size,stride,padding),
            MixConv2d(in_planes,out_planes),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )
    def forward(self,x):
        return self.m(x)

class ConvBNRelu(nn.Module):
    def __init__(self,in_planes, out_planes, kernel_size=3, stride=1,padding=0):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size,stride,padding),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )
    def forward(self,x):
        return self.m(x)

class ConvBN(nn.Module):
    def __init__(self,in_planes, out_planes, kernel_size=3, stride=1,padding=0):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size,stride,padding),
            nn.BatchNorm2d(out_planes)
        )
    def forward(self,x):
        return self.m(x)


class SepConv(nn.Module):
    def __init__(self,in_planes, out_planes, kernel_size=3, stride=1,padding=1):
        super().__init__()
        self.m = nn.Sequential(
            DwConvBNRelu(in_planes,in_planes,kernel_size,stride,padding),

            ConvBNRelu(in_planes,out_planes,1)
        )
    def forward(self,x):
        return self.m(x)

class MBConv6(nn.Module):
    def __init__(self,in_planes, out_planes, kernel_size=3, stride=1,padding=1,downsample=None):
        super().__init__()
        self.m = nn.Sequential(
            ConvBNRelu(in_planes,in_planes*6,1),

            DwConvBNRelu(in_planes*6,in_planes*6,kernel_size,stride,padding),

            ConvBN(in_planes*6, out_planes, 1)
        )

        self.downsample = downsample

    def forward(self,x):
        if self.downsample:
            return F.relu(self.downsample(x)+self.m(x))
        else:
            return F.relu(x+self.m(x))

class MBConv3(nn.Module):
    def __init__(self,in_planes, out_planes, kernel_size=3, stride=1,padding=1,downsample=None):
        super().__init__()
        self.m = nn.Sequential(
            ConvBNRelu(in_planes,in_planes*3,1),

            DwConvBNRelu(in_planes*3, in_planes*3, kernel_size, stride, padding),

            SeBlock(in_planes*3),

            ConvBN(in_planes*3, out_planes, 1)
        )

        self.downsample = downsample

    def forward(self, x):
        if self.downsample:
            return F.relu(self.downsample(x) + self.m(x))
        else:
            return F.relu(x + self.m(x))


class MnasNetA1(nn.Module):
    def __init__(self,num_classes=1000,dropout=0.2):
        super().__init__()

        self.feature = nn.Sequential(
            ConvBNRelu(3,32,3,2,1),
            SepConv(32,16,3,1,1),

            MBConv6(16,16,3),
            MBConv6(16,16,3),
            ConvBNRelu(16,24,3,2,1),
            # ConvBNRelu(16, 24, 1, 2, 0),

            MBConv3(24,24,5,padding=2),
            SeBlock(24),
            MBConv3(24, 24, 5, padding=2),
            SeBlock(24),
            MBConv3(24, 24, 5, padding=2),
            SeBlock(24),
            ConvBNRelu(24, 40, 3, 2, 1),
            # ConvBNRelu(24, 40, 1, 2, 0),

            MBConv6(40, 40, 3),
            MBConv6(40, 40, 3),
            MBConv6(40, 40, 3),
            MBConv6(40, 40, 3),
            ConvBNRelu(40, 80, 3, 2, 1),
            # ConvBNRelu(40, 80, 1, 2, 0),

            MBConv6(80, 80, 3),
            SeBlock(80),
            MBConv6(80, 80, 3),
            SeBlock(80),
            ConvBNRelu(80, 112, 3, 1, 1),
            # ConvBNRelu(80, 112, 1, 1, 0),

            MBConv6(112, 112, 5,padding=2),
            SeBlock(112),
            MBConv6(112, 112, 5, padding=2),
            SeBlock(112),
            MBConv6(112, 112, 5, padding=2),
            SeBlock(112),
            ConvBNRelu(112, 160, 3, 2, 1),
            # ConvBNRelu(112, 160, 1, 2, 0),

            MBConv6(160, 160, 3),
            ConvBNRelu(160, 320, 3, 1, 1)
            # ConvBNRelu(160, 320, 1, 1, 0)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                nn.Linear(320, num_classes))

        _initialize_weights(self)

    def forward(self, x):
        x = self.feature(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    x = torch.randn([5, 3, 224, 224])
    m = MnasNetA1(num_classes=10)
    print(m)
    print(m(x).shape)
    torch.save(m.state_dict(), "m.pt")