from torch import nn
import torch
from torch.nn import functional as F
from collections import OrderedDict

from .resnet import CBR,Stem,Classify
from .common import Flatten,CBAM


class CSPCatBlock(nn.Module):
    def __init__(self, in_c,hide_c,out_c):
        super().__init__()
        t_c = in_c//2
        self.m1 = nn.Sequential(
            CBR(t_c,hide_c,1),
            nn.Sequential(
                CBR(t_c,hide_c,1),
                CBR(hide_c,hide_c,3),
            ),
            nn.Sequential(
                CBR(t_c, hide_c, 1),
                # CBR(hide_c, hide_c, 3),
                # CBR(hide_c, hide_c, 3),
                CBR(hide_c, hide_c, 3, dilation=2)
            )
        )
        self.m1_conv1x1=CBR(hide_c*3,t_c,1)

        self.m2 = nn.Sequential(
            CBR(t_c,hide_c,1),
            nn.Sequential(
                CBR(t_c,hide_c,1),
                CBR(hide_c,hide_c,3),
            ),
            nn.Sequential(
                CBR(t_c, hide_c, 1),
                # CBR(hide_c, hide_c, 3),
                # CBR(hide_c, hide_c, 3),
                CBR(hide_c, hide_c, 3, dilation=2)
            )
        )

        self.m2_conv1x1 = CBR(hide_c * 3, t_c, 1)

        self.conv1x1 = CBR(in_c, out_c, 1,act=False)

    def forward(self,x):
        x1,x2 = torch.chunk(x,2,1) # 按通道拆分

        x1 = torch.cat((self.m1[0](x1),self.m1[1](x1),self.m1[2](x1)),1)
        x1 = self.m1_conv1x1(x1)

        x2 = torch.cat((self.m2[0](x2), self.m2[1](x2), self.m2[2](x2)), 1)
        x2 = self.m2_conv1x1(x2)

        x12 = torch.cat((x1,x2),1)
        x12 = self.conv1x1(x12)

        return F.relu(x+x12)

class CSPCatBlockv2(nn.Module):
    def __init__(self, in_c,hide_c,out_c):
        super().__init__()
        t_c = in_c//4
        self.m = nn.Sequential(
            CBR(t_c,hide_c,1),
            nn.Sequential(
                CBR(t_c,hide_c,1),
                CBR(hide_c,hide_c,3),
            ),
            nn.Sequential(
                CBR(t_c, hide_c, 1),
                # CBR(hide_c, hide_c, 3),
                # CBR(hide_c, hide_c, 3),
                CBR(hide_c,hide_c,3,dilation=2)
            ),

        nn.Sequential(
            CBR(t_c, hide_c, 1),
            CBR(hide_c, hide_c, (1,3),padding=(0,1)),
            CBR(hide_c, hide_c, (3,1),padding=(1,0)),
        ))

        self.conv1x1 = CBR(in_c, out_c, 1,act=False)

        self.seblock = CBAM(out_c,3,out_c//4)

    def forward(self,x):
        x1,x2,x3,x4 = torch.chunk(x,4,1) # 按通道拆分

        x1234 = torch.cat((self.m[0](x1), self.m[1](x2), self.m[2](x3), self.m[3](x4)), 1)
        x1234 = self.conv1x1(x1234)
        x1234 = self.seblock(x1234)
        return F.relu(x+x1234)


class CSPCatBottleBlockv2(nn.Module):
    def __init__(self, in_c,hide_c,out_c):
        super().__init__()
        t_c = in_c//4
        self.m = nn.Sequential(
            CBR(t_c,t_c,1),
            nn.Sequential(
                CBR(t_c,hide_c,1),
                CBR(hide_c,hide_c,3),
                CBR(hide_c,t_c,1),
            ),
            nn.Sequential(
                CBR(t_c, hide_c, 1),
                # CBR(hide_c, hide_c, 3),
                # CBR(hide_c, hide_c, 3),
                CBR(hide_c, hide_c, 3, dilation=2),
                CBR(hide_c, t_c, 1),
            ),

        nn.Sequential(
            CBR(t_c, hide_c, 1),
            CBR(hide_c, hide_c, (1,3),padding=(0,1)),
            CBR(hide_c, hide_c, (3,1),padding=(1,0)),
            CBR(hide_c, t_c, 1),
        ))

        self.conv1x1 = CBR(in_c, out_c, 1,act=False)

        self.seblock = CBAM(out_c, 3, out_c // 4)

    def forward(self,x):
        x1,x2,x3,x4 = torch.chunk(x,4,1) # 按通道拆分

        x1234 = torch.cat((self.m[0](x1), self.m[1](x2), self.m[2](x3), self.m[3](x4)), 1)
        x1234 = self.conv1x1(x1234)
        x1234 = self.seblock(x1234)
        return F.relu(x+x1234)

class Downsample(nn.Module):
    def __init__(self, in_c,hide_c,out_c):
        super().__init__()
        self.m = nn.Sequential(
            nn.MaxPool2d(3,2,1),
            nn.Sequential(CBR(in_c,hide_c,1),CBR(hide_c,hide_c,3,2,1)),
            nn.Sequential(CBR(in_c, hide_c, 1), CBR(hide_c, hide_c, 3, 2, 1,groups=hide_c),CBR(hide_c,hide_c,1)),
            # nn.Sequential(CBR(in_c, hide_c, 1),CBR(hide_c, hide_c, 3, 1, 1), CBR(hide_c, hide_c, 3, 2, 1)),
            nn.Sequential(CBR(in_c, hide_c, 1),CBR(hide_c, hide_c, 3, 2, dilation=2)),
            nn.Sequential(CBR(in_c, hide_c, 1),CBR(hide_c, hide_c, (1,3), (1,2), (0,1)), CBR(hide_c, hide_c, (3,1), (2,1), (1,0))),
        )
        self.conv1x1 = CBR(in_c,in_c,1)

    def forward(self,x):
        x1 = self.m[0](x)
        x2 = self.conv1x1(torch.cat((self.m[1](x),self.m[2](x),self.m[3](x),self.m[4](x)),1))

        return torch.cat((x1,x2),1)

class Resnet18(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = Stem(in_c,64,7,2)
        self.layer1 = nn.Sequential(
            *[CSPCatBlockv2(64,16,64) for _ in range(4)],
        )
        self.layer2 = nn.Sequential(
            Downsample(64,16,128),
            *[CSPCatBlockv2(128, 32, 128) for _ in range(4)],
        )
        self.layer3 = nn.Sequential(
            Downsample(128, 32, 256),
            *[CSPCatBlockv2(256,64,256) for _ in range(4)],
        )
        self.layer4 = nn.Sequential(
            Downsample(256, 64, 512),
            *[CSPCatBlockv2(512,128,512) for _ in range(4)],
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

class Resnet18v2(nn.Module):
    def __init__(self,in_c=3,num_classes=1000):
        super().__init__()
        self.stem = Stem(in_c,64,7,2)
        self.layer1 = nn.Sequential(
            *[CSPCatBottleBlockv2(64,8,64) for _ in range(4)],
        )
        self.layer2 = nn.Sequential(
            Downsample(64,16,128),
            *[CSPCatBottleBlockv2(128, 16, 128) for _ in range(4)],
        )
        self.layer3 = nn.Sequential(
            Downsample(128, 32, 256),
            *[CSPCatBottleBlockv2(256,32,256) for _ in range(4)],
        )
        self.layer4 = nn.Sequential(
            Downsample(256, 64, 512),
            *[CSPCatBottleBlockv2(512,64,512) for _ in range(4)],
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

if __name__ == "__main__":
    x = torch.rand([2,3,224,224])
    m = Resnet18v2()
    # _initParmas(m.modules())
    print(m(x).shape)
    torch.save(m.state_dict(),'Resnet18.pth')