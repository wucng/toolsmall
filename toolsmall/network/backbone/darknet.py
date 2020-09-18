"""
https://zhuanlan.zhihu.com/p/105278156
"""
import torch
from torch import nn
from .moudle import MixConv2d,DwConv2dV1,DwConv2dV3,\
    DilatedConv2d,DwConv2dV2,SeBlock,ShuffleBlockV1,\
    _initialize_weights

class Conv_BN_LeakyReLU2(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU2, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            # MixConv2d(in_channels,out_channels,stride=stride),
            # DwConv2dV1(in_channels,out_channels,ksize,stride,padding),
            # DwConv2dV2(in_channels,out_channels,ksize,stride,padding),
            # DwConv2dV3(in_channels,out_channels,ksize,stride,padding),

            # DilatedConv2d(in_channels,out_channels,ksize,stride,(ksize+2*(2-1))//2,dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)

class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            MixConv2d(in_channels,out_channels,stride=stride),
            # DwConv2dV1(in_channels,out_channels,ksize,stride,padding),
            # DwConv2dV2(in_channels,out_channels,ksize,stride,padding),
            # DwConv2dV3(in_channels,out_channels,ksize,stride,padding),

            # DilatedConv2d(in_channels,out_channels,ksize,stride,(ksize+2*(2-1))//2,dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class DarkNet_19(nn.Module):
    def __init__(self, num_classes=1000):
        print("Initializing the darknet19 network ......")

        super(DarkNet_19, self).__init__()
        # backbone network : DarkNet-19
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU2(3, 32, 3, 1),
            nn.MaxPool2d((2, 2), 2),
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU2(128, 64, 1),
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 16, c = 256
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU2(256, 128, 1),
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 32, c = 512
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU2(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU2(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 32, c = 1024
        self.conv_6 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU2(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU2(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1)
        )

        self.conv_7 = nn.Conv2d(1024, num_classes, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        _initialize_weights(self)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)

        x = self.conv_7(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class resblock2(nn.Module):
    def __init__(self, ch, nblocks=1):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.Sequential(
                Conv_BN_LeakyReLU2(ch, ch//2, 1),
                Conv_BN_LeakyReLU(ch//2, ch, 3, padding=1)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x

"""结合senet,shufflenet"""
class resblock(nn.Module):
    def __init__(self, ch, nblocks=1):
        super().__init__()
        self.module_list = nn.ModuleList()
        self.module_se = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.Sequential(
                Conv_BN_LeakyReLU2(ch, ch//2, 1),
                ShuffleBlockV1(4),
                Conv_BN_LeakyReLU(ch//2, ch, 3, padding=1)
            )
            self.module_list.append(resblock_one)
            self.module_se.append(SeBlock(ch))

    def forward(self, x):
        for i,module in enumerate(self.module_list):
            x = self.module_se[i](module(x)) + x
        return x

"""DarkNet_53简化而来"""
class DarkNet_33(nn.Module):
    def __init__(self, num_classes=1000,dropout=0.2):
        print("Initializing the darknet33 network ......")
        super(DarkNet_33, self).__init__()
        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv_BN_LeakyReLU2(3, 32, 3, padding=1),
            Conv_BN_LeakyReLU(32, 64, 3, padding=1, stride=2),
            resblock(64, nblocks=1)
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, padding=1, stride=2),
            resblock(128, nblocks=2)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, padding=1, stride=2),
            resblock(256, nblocks=4)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, padding=1, stride=2),
            resblock(512, nblocks=4)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, padding=1, stride=2),
            resblock(1024, nblocks=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                nn.Linear(1024, num_classes))

        _initialize_weights(self)

    def forward(self, x, targets=None):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# darknet-53 code
class DarkNet_53(nn.Module):
    def __init__(self, num_classes=1000,dropout=0.2):
        print("Initializing the darknet53 network ......")
        super(DarkNet_53, self).__init__()
        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv_BN_LeakyReLU2(3, 32, 3, padding=1),
            Conv_BN_LeakyReLU(32, 64, 3, padding=1, stride=2),
            resblock(64, nblocks=1)
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, padding=1, stride=2),
            resblock(128, nblocks=2)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, padding=1, stride=2),
            resblock(256, nblocks=8)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, padding=1, stride=2),
            resblock(512, nblocks=8)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, padding=1, stride=2),
            resblock(1024, nblocks=4)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                nn.Linear(1024, num_classes))


        _initialize_weights(self)

    def forward(self, x, targets=None):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    x = torch.randn([5, 3, 224, 224])
    m = DarkNet_33(num_classes=10)
    print(m)
    print(m(x).shape)
    torch.save(m.state_dict(), "m.pt")