from torch import nn
import torch
from torch.nn import functional as F

from .resnet import CBR,Classify
from .common import Flatten


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

        self.classifier = Classify(512,num_classes)

    def forward(self,x):
        x = self.feature(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    x = torch.rand([2,3,224,224])
    m = VGG16BNV2()
    # _initParmas(m.modules())
    print(m(x).shape)
    torch.save(m.state_dict(),'VGG16BNV2.pth')