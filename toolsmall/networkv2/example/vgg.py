from torch import nn
import torch
from torch.nn import functional as F

from .resnet import CBR,Classify
from .common import Flatten
from .darknet import CBL,SPP


class VGG16BN(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.feature = nn.Sequential(
            CBR(in_c,64,3),
            CBR(64,64,3),
            nn.MaxPool2d(2,2),

            CBR(64, 128, 3),
            CBR(128, 128, 3),
            nn.MaxPool2d(2, 2),

            CBR(128, 256, 3),
            CBR(256, 256, 3),
            CBR(256, 256, 3),
            nn.MaxPool2d(2, 2),

            CBR(256, 512, 3),
            CBR(512, 512, 3),
            CBR(512, 512, 3),
            nn.MaxPool2d(2, 2),

            CBR(512, 512, 3),
            CBR(512, 512, 3),
            CBR(512, 512, 3),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self,x):
        x = self.feature(x)
        x = self.classifier(x)
        return x

class VGG16BNV2(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()

        self.stem = nn.Sequential(
            CBR(in_c, 64, 3),
            CBR(64, 64, 3),
            nn.MaxPool2d(2, 2),
        )
        self.layer1 = nn.Sequential(
            CBR(64, 128, 3),
            CBR(128, 128, 3),
            nn.MaxPool2d(2, 2),
        )
        self.layer2 = nn.Sequential(
            CBR(128, 256, 3),
            CBR(256, 256, 3),
            CBR(256, 256, 3),
            nn.MaxPool2d(2, 2),
        )
        self.layer3 = nn.Sequential(
            CBR(256, 512, 3),
            CBR(512, 512, 3),
            CBR(512, 512, 3),
            nn.MaxPool2d(2, 2),
        )
        self.layer4 = nn.Sequential(
            CBR(512, 512, 3),
            CBR(512, 512, 3),
            CBR(512, 512, 3),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = Classify(512,num_classes)

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x


class VGG16BNV3(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()

        self.stem = nn.Sequential(
            CBR(in_c, 64, 3),
            CBR(64, 64, 3,2),
        )
        self.layer1 = nn.Sequential(
            CBR(64, 128, 3,2),
            CBR(128, 128, 3,groups=64),
        )
        self.layer2 = nn.Sequential(
            CBR(128, 256, 3,2),
            CBR(256, 256, 3,groups=128),
            CBR(256, 256, 3,groups=128),
        )
        self.layer3 = nn.Sequential(
            CBR(256, 512, 3,2),
            CBR(512, 512, 3,groups=256),
            CBR(512, 512, 3,groups=256),
        )
        self.layer4 = nn.Sequential(
            CBR(512, 512, 3,2),
            CBR(512, 512, 3,groups=256),
            CBR(512, 512, 3,groups=256),
        )

        self.classifier = Classify(512,num_classes)

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x

class Yolov4One(nn.Module):
    def __init__(self,num_classes=21,num_anchor=3):
        super().__init__()
        _m = VGG16BNV3()
        self.backbone=nn.Sequential(
            *[_m.stem,_m.layer1,_m.layer2,_m.layer3,_m.layer4]
        )

        self.layer1 = nn.Sequential(
            CBL(512,256,1),
            CBL(256,512,3),
            CBL(512,256,1),
            SPP(256,1024),
            CBL(1024,256,1),
            CBL(256,512,3),
            CBL(512,256,1),
        )

        self.layer11 = nn.Sequential(
            CBL(256, 128, 1),
            nn.Upsample(scale_factor=2),

            CBL(512, 256, 1),

            CBL(384, 128, 1),
            CBL(128, 256, 3),
            CBL(256, 128, 1),
            CBL(128, 256, 3),
            CBL(256, 128, 1),
        )

        self.layer12 = nn.Sequential(
            CBL(128, 64, 1),
            nn.Upsample(scale_factor=2),

            CBL(256, 128, 1),

            CBL(192, 64, 1),
            CBL(64, 128, 3),
            CBL(128, 64, 1),
            CBL(64, 128, 3),
            CBL(128, 64, 1),
        )

        self.layer13 = nn.Sequential(
            CBL(64, 128, 3),
            nn.Conv2d(128, num_anchor * (num_classes + 5), 1)
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

        return x3

if __name__ == "__main__":
    x = torch.rand([2,3,224,224])
    m = VGG16BNV2()
    # _initParmas(m.modules())
    print(m(x).shape)
    torch.save(m.state_dict(),'VGG16BNV2.pth')