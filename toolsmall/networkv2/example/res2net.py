from torch import nn
import torch
from torch.nn import functional as F

from .resnet import Stem,Classify,Downsample,CBR
from .common import SEblock,SKblockv2,CBAM,SAM,channel_shuffle,Flatten

class BottleneckS1(nn.Module):
    def __init__(self,in_c,hide_c,out_c,ksize=3,stride=1,groups=4):
        super().__init__()
        tmp = hide_c//groups
        self.conv1x1_1 = CBR(in_c,hide_c,1)
        self.conv1x1_2 = CBR(hide_c,out_c,1)
        self.conv3x3_1 = CBR(tmp,tmp,3)
        self.conv3x3_2 = CBR(tmp,tmp,3)
        self.conv3x3_3 = CBR(tmp,tmp,3)
        # self.seblock = SEblock(out_c,out_c//4,out_c)
        # self.seblock = SAM(out_c)
        self.seblock = CBAM(out_c, 3, out_c // 4)

        self.downsample = None
        if stride > 1 or in_c != out_c:
            self.downsample = Downsample(in_c, out_c, 1, stride)

    def forward2(self, x):
        _x = self.conv1x1_1(x)
        x1,x2,x3,x4 = torch.chunk(_x,4,1)
        x2 = self.conv3x3_1(x2)
        x3 = self.conv3x3_2(x2+x3)
        x4 = self.conv3x3_3(x3+x4)

        _x = torch.cat((x1,x2,x3,x4),1)
        _x = self.conv1x1_2(_x)
        _x = self.seblock(_x)

        return F.relu(_x+x if self.downsample is None else self.downsample(x))

    def forward(self, x):
        _x = self.conv1x1_1(x)
        x1,x2,x3,x4 = torch.chunk(_x,4,1)
        x2 = self.conv3x3_1(x1+x2)
        x3 = self.conv3x3_2(x1+x2+x3)
        x4 = self.conv3x3_3(x1+x2+x3+x4)

        _x = torch.cat((x1, x2, x3, x4), 1)
        _x = channel_shuffle(_x,4)+_x
        _x = self.conv1x1_2(_x)
        _x = self.seblock(_x)

        return F.relu(_x+x if self.downsample is None else self.downsample(x))

class BottleneckS1_DW(nn.Module):
    def __init__(self,in_c,hide_c,out_c,ksize=3,stride=1,groups=4):
        super().__init__()
        tmp = hide_c//groups
        self.conv1x1_1 = CBR(in_c,hide_c,1)
        self.conv1x1_2 = CBR(hide_c,out_c,1)
        self.conv3x3_1 = CBR(tmp,tmp,3,groups=tmp)
        self.conv3x3_2 = CBR(tmp,tmp,3,groups=tmp)
        self.conv3x3_3 = CBR(tmp,tmp,3,groups=tmp)
        # self.seblock = SEblock(out_c,out_c//4,out_c)
        # self.seblock = SAM(out_c)
        self.seblock = CBAM(out_c, 3, out_c // 4)

        self.downsample = None
        if stride > 1 or in_c != out_c:
            self.downsample = Downsample(in_c, out_c, 1, stride)

    def forward2(self, x):
        _x = self.conv1x1_1(x)
        x1,x2,x3,x4 = torch.chunk(_x,4,1)
        x2 = self.conv3x3_1(x2)
        x3 = self.conv3x3_2(x2+x3)
        x4 = self.conv3x3_3(x3+x4)

        _x = torch.cat((x1, x2, x3, x4), 1)
        _x = channel_shuffle(_x,4)+_x
        _x = self.conv1x1_2(_x)
        _x = self.seblock(_x)

        return F.relu(_x+x if self.downsample is None else self.downsample(x))

    def forward(self, x):
        _x = self.conv1x1_1(x)
        x1,x2,x3,x4 = torch.chunk(_x,4,1)
        x2 = self.conv3x3_1(x1+x2)
        x3 = self.conv3x3_2(x1+x2+x3)
        x4 = self.conv3x3_3(x1+x2+x3+x4)

        _x = torch.cat((x1,x2,x3,x4),1)+_x
        _x = self.conv1x1_2(_x)
        _x = self.seblock(_x)

        return F.relu(_x+x if self.downsample is None else self.downsample(x))

class BottleneckS2(nn.Module):
    def __init__(self,in_c,hide_c,out_c,ksize=3,stride=2):
        super().__init__()
        self.m = nn.Sequential(
            CBR(in_c, hide_c, 1, 1, bias=False),
            CBR(hide_c, hide_c, ksize, stride, padding=ksize // 2),
            nn.Conv2d(hide_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            # SEblock(out_c, out_c // 4, out_c),  # 只是增加这个模块
            CBAM(out_c,3,out_c//4)
        )
        self.downsample = None
        if stride > 1 or in_c!=out_c:
            self.downsample = Downsample(in_c, out_c,1,stride)

    def forward(self, x):
        return F.relu(self.m(x) + x if self.downsample is None else self.downsample(x))

class BottleneckS2_DW(nn.Module):
    def __init__(self,in_c,hide_c,out_c,ksize=3,stride=2):
        super().__init__()
        self.m = nn.Sequential(
            CBR(in_c, hide_c, 1, 1, bias=False),
            CBR(hide_c, hide_c, ksize, stride, padding=ksize // 2,groups=hide_c),
            nn.Conv2d(hide_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            # SEblock(out_c, out_c // 4, out_c),  # 只是增加这个模块
            CBAM(out_c,3,out_c//4)
        )
        self.downsample = None
        if stride > 1 or in_c!=out_c:
            self.downsample = Downsample(in_c, out_c,1,stride)

    def forward(self, x):
        return F.relu(self.m(x) + x if self.downsample is None else self.downsample(x))

class Resnet50(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = Stem(in_c,64,7,2)
        self.layer1 = nn.Sequential(
            BottleneckS1(64,64,256,3,1),
            BottleneckS1(256,64,256,3,1),
            BottleneckS1(256,64,256,3,1),
        )
        self.layer2 = nn.Sequential(
            BottleneckS2(256, 128, 512, 3, 2),
            BottleneckS1(512, 128, 512, 3, 1),
            BottleneckS1(512, 128, 512, 3, 1),
            BottleneckS1(512, 128, 512, 3, 1),
        )
        self.layer3 = nn.Sequential(
            BottleneckS2(512, 256, 1024, 3, 2),
            BottleneckS1(1024, 256, 1024, 3, 1),
            BottleneckS1(1024, 256, 1024, 3, 1),
            BottleneckS1(1024, 256, 1024, 3, 1),
            BottleneckS1(1024, 256, 1024, 3, 1),
            BottleneckS1(1024, 256, 1024, 3, 1),
        )
        self.layer4 = nn.Sequential(
            BottleneckS2(1024, 512, 2048, 3, 2),
            BottleneckS1(2048, 512, 2048, 3, 1),
            BottleneckS1(2048, 512, 2048, 3, 1)
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

class Resnet38(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = Stem(in_c,64,7,2)
        self.layer1 = nn.Sequential(
            BottleneckS1(64,64,256,3,1),
            BottleneckS1(256,64,256,3,1),
            # BottleneckS1(256,64,256,3,1),
        )
        self.layer2 = nn.Sequential(
            BottleneckS2(256, 128, 512, 3, 2),
            BottleneckS1(512, 128, 512, 3, 1),
            BottleneckS1(512, 128, 512, 3, 1),
            # BottleneckS1(512, 128, 512, 3, 1),
        )
        self.layer3 = nn.Sequential(
            BottleneckS2(512, 256, 1024, 3, 2),
            BottleneckS1(1024, 256, 1024, 3, 1),
            BottleneckS1(1024, 256, 1024, 3, 1),
            BottleneckS1(1024, 256, 1024, 3, 1),
            # BottleneckS1(1024, 256, 1024, 3, 1),
            # BottleneckS1(1024, 256, 1024, 3, 1),
        )
        self.layer4 = nn.Sequential(
            BottleneckS2(1024, 512, 2048, 3, 2),
            BottleneckS1(2048, 512, 2048, 3, 1),
            BottleneckS1(2048, 512, 2048, 3, 1)
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

class Resnet29(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = Stem(in_c,64,7,2)
        self.layer1 = nn.Sequential(
            BottleneckS1(64,64,256,3,1),
            BottleneckS1(256,64,256,3,1),
            # BottleneckS1(256,64,256,3,1),
        )
        self.layer2 = nn.Sequential(
            BottleneckS2(256, 128, 512, 3, 2),
            BottleneckS1(512, 128, 512, 3, 1),
            # BottleneckS1(512, 128, 512, 3, 1),
            # BottleneckS1(512, 128, 512, 3, 1),
        )
        self.layer3 = nn.Sequential(
            BottleneckS2(512, 256, 1024, 3, 2),
            BottleneckS1(1024, 256, 1024, 3, 1),
            BottleneckS1(1024, 256, 1024, 3, 1),
            # BottleneckS1(1024, 256, 1024, 3, 1),
            # BottleneckS1(1024, 256, 1024, 3, 1),
            # BottleneckS1(1024, 256, 1024, 3, 1),
        )
        self.layer4 = nn.Sequential(
            BottleneckS2(1024, 512, 2048, 3, 2),
            BottleneckS1(2048, 512, 2048, 3, 1),
            # BottleneckS1(2048, 512, 2048, 3, 1)
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


class Resnet50_DW(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = Stem(in_c,64,7,2)
        self.layer1 = nn.Sequential(
            BottleneckS1_DW(64,64,256,3,1),
            BottleneckS1_DW(256,64,256,3,1),
            BottleneckS1_DW(256,64,256,3,1),
        )
        self.layer2 = nn.Sequential(
            BottleneckS2_DW(256, 128, 512, 3, 2),
            BottleneckS1_DW(512, 128, 512, 3, 1),
            BottleneckS1_DW(512, 128, 512, 3, 1),
            BottleneckS1_DW(512, 128, 512, 3, 1),
        )
        self.layer3 = nn.Sequential(
            BottleneckS2_DW(512, 256, 1024, 3, 2),
            BottleneckS1_DW(1024, 256, 1024, 3, 1),
            BottleneckS1_DW(1024, 256, 1024, 3, 1),
            BottleneckS1_DW(1024, 256, 1024, 3, 1),
            BottleneckS1_DW(1024, 256, 1024, 3, 1),
            BottleneckS1_DW(1024, 256, 1024, 3, 1),
        )
        self.layer4 = nn.Sequential(
            BottleneckS2_DW(1024, 512, 2048, 3, 2),
            BottleneckS1_DW(2048, 512, 2048, 3, 1),
            BottleneckS1_DW(2048, 512, 2048, 3, 1)
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

class Resnet38_DW(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = Stem(in_c,64,7,2)
        self.layer1 = nn.Sequential(
            BottleneckS1_DW(64,64,256,3,1),
            BottleneckS1_DW(256,64,256,3,1),
            # BottleneckS1_DW(256,64,256,3,1),
        )
        self.layer2 = nn.Sequential(
            BottleneckS2_DW(256, 128, 512, 3, 2),
            BottleneckS1_DW(512, 128, 512, 3, 1),
            BottleneckS1_DW(512, 128, 512, 3, 1),
            # BottleneckS1_DW(512, 128, 512, 3, 1),
        )
        self.layer3 = nn.Sequential(
            BottleneckS2_DW(512, 256, 1024, 3, 2),
            BottleneckS1_DW(1024, 256, 1024, 3, 1),
            BottleneckS1_DW(1024, 256, 1024, 3, 1),
            BottleneckS1_DW(1024, 256, 1024, 3, 1),
            # BottleneckS1_DW(1024, 256, 1024, 3, 1),
            # BottleneckS1_DW(1024, 256, 1024, 3, 1),
        )
        self.layer4 = nn.Sequential(
            BottleneckS2_DW(1024, 512, 2048, 3, 2),
            BottleneckS1_DW(2048, 512, 2048, 3, 1),
            BottleneckS1_DW(2048, 512, 2048, 3, 1)
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

class Resnet26_DW(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = Stem(in_c,64,7,2)
        self.layer1 = nn.Sequential(
            BottleneckS1_DW(64,64,256,3,1),
            BottleneckS1_DW(256,64,256,3,1),
            # BottleneckS1_DW(256,64,256,3,1),
        )
        self.layer2 = nn.Sequential(
            BottleneckS2_DW(256, 128, 512, 3, 2),
            BottleneckS1_DW(512, 128, 512, 3, 1),
            # BottleneckS1_DW(512, 128, 512, 3, 1),
            # BottleneckS1_DW(512, 128, 512, 3, 1),
        )
        self.layer3 = nn.Sequential(
            BottleneckS2_DW(512, 256, 1024, 3, 2),
            BottleneckS1_DW(1024, 256, 1024, 3, 1),
            # BottleneckS1_DW(1024, 256, 1024, 3, 1),
            # BottleneckS1_DW(1024, 256, 1024, 3, 1),
            # BottleneckS1_DW(1024, 256, 1024, 3, 1),
            # BottleneckS1_DW(1024, 256, 1024, 3, 1),
        )
        self.layer4 = nn.Sequential(
            BottleneckS2_DW(1024, 512, 2048, 3, 2),
            BottleneckS1_DW(2048, 512, 2048, 3, 1),
            # BottleneckS1_DW(2048, 512, 2048, 3, 1)
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
    m = Resnet26_DW()
    # _initParmas(m.modules())
    print(m(x).shape)
    torch.save(m.state_dict(),'Resnet38_DW.pth')