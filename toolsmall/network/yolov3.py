"""
参考：
yolov3
https://blog.csdn.net/dz4543/article/details/90049377

yolov3-spp
https://blog.csdn.net/qq_33270279/article/details/103898245
https://blog.csdn.net/qq_39056987/article/details/104327638

yolov4
https://blog.csdn.net/justsolow/article/details/106401065
"""
import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
# from utils.layers import *


class Flatten(nn.Module):
    """
    :param
    :return

    :example
        x = torch.rand([3,1000,1,1]);
        x = Flatten()(x);
        print(x.shape) # [3,1000]
    """
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return torch.flatten(x,1)

def convBNLelu(in_channels=3, out_channels=32,
               kernel_size=3, stride=1, padding=1,
               groups=1, bias=False,negative_slope=0.1):
    return nn.Sequential(OrderedDict([
            ('Conv2d',nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)),
            ('BatchNorm2d', nn.BatchNorm2d(out_channels, momentum=0.03, eps=1E-4)),
            ('activation', nn.LeakyReLU(negative_slope, inplace=True))
        ]))


class ResidualBlock(nn.Module):
    def __init__(self,in_channels=64, out_channels=32):
        super().__init__()
        self.residual=nn.Sequential(
        convBNLelu(in_channels,out_channels,1,1,0),
        convBNLelu(out_channels,in_channels,3,1,1)
        )

    def forward(self,x):
        return x+self.residual(x)

class SPPNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.spp = nn.Sequential(
            nn.MaxPool2d(1,1,0),
            nn.MaxPool2d(5,1,2),
            nn.MaxPool2d(9,1,4),
            nn.MaxPool2d(13,1,6)
        )

    def forward(self,x):
        x1 = self.spp[0](x)
        x2 = self.spp[1](x)
        x3 = self.spp[2](x)
        x4 = self.spp[3](x)

        return x1+x2+x3+x4

class SPPNet(nn.Module):
    """论文的版本"""
    def __init__(self,useAdd=False):
        super().__init__()
        self.useAdd = useAdd
        self.spp = nn.Sequential(
            nn.MaxPool2d(1,1,0),
            nn.MaxPool2d(5,1,2),
            nn.MaxPool2d(9,1,4),
            nn.MaxPool2d(13,1,6)
        )

    def forward(self,x):
        bs,c,h,w = x.shape
        c1 = c//4
        x1 = self.spp[0](x[:,:c1,...])
        x2 = self.spp[1](x[:,c1:c1*2,...])
        x3 = self.spp[2](x[:,c1*2:c1*3,...])
        x4 = self.spp[3](x[:,c1*3:,...])

        if self.useAdd:
            return torch.cat((x1,x2,x3,x4),1)+x
        else:
            return torch.cat((x1,x2,x3,x4),1)

class ASPP(nn.Module):
    """https://blog.csdn.net/justsolow/article/details/106401065#t3
    """
    def __init__(self,in_channels,useAdd=False):
        super().__init__()
        self.useAdd = useAdd
        out_channels = in_channels//4
        assert out_channels*4==in_channels
        self.aspp = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,3,1,6,6), # 3+2*(6-1)
            nn.Conv2d(out_channels,out_channels,3,1,12,12),
            nn.Conv2d(out_channels,out_channels,3,1,18,18),
            nn.Conv2d(out_channels,out_channels,3,1,24,24)
        )

    def forward(self,x):
        bs, c, h, w = x.shape
        c1 = c // 4
        x1 = self.aspp[0](x[:, :c1, ...])
        x2 = self.aspp[1](x[:, c1:c1 * 2, ...])
        x3 = self.aspp[2](x[:, c1 * 2:c1 * 3, ...])
        x4 = self.aspp[3](x[:, c1 * 3:, ...])

        if self.useAdd:
            return torch.cat((x1, x2, x3, x4), 1) + x
        else:
            return torch.cat((x1, x2, x3, x4), 1)

class RFB(nn.Module):
    def __init__(self,in_channels,useAdd=False):
        super().__init__()
        self.useAdd = useAdd
        out_channels = in_channels // 4
        assert out_channels * 4 == in_channels

        self.rfb = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,1,1,0),
            nn.Conv2d(out_channels,out_channels,3,1,1,1),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 3, 3),

            nn.Conv2d(out_channels, out_channels, 5, 1, 2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 5, 5),
        )

    def forward(self,x):
        bs, c, h, w = x.shape
        c1 = c // 4
        x1 = x[:, :c1, ...]
        x2 = self.rfb[0:2](x[:, c1:c1 * 2, ...])
        x3 = self.rfb[2:4](x[:, c1 * 2:c1 * 3, ...])
        x4 = self.rfb[4:](x[:, c1 * 3:, ...])

        if self.useAdd:
            return torch.cat((x1, x2, x3, x4), 1) + x
        else:
            return torch.cat((x1, x2, x3, x4), 1)


class Darknet53(nn.Module):
    def __init__(self,num_classes=1000,pretrained=False):
        super().__init__()
        self.model=nn.Sequential(
            convBNLelu(3,32,3,1,1),
            convBNLelu(32,64,3,2,1),
            ResidualBlock(64,32), # id =2

            convBNLelu(64,128,3,2,1),
            ResidualBlock(128, 64),
            ResidualBlock(128, 64), # s4 id=5

            convBNLelu(128, 256, 3, 2, 1),
            ResidualBlock(256, 128),
            ResidualBlock(256, 128),
            ResidualBlock(256, 128),
            ResidualBlock(256, 128),
            ResidualBlock(256, 128),
            ResidualBlock(256, 128),
            ResidualBlock(256, 128),
            ResidualBlock(256, 128), # s8 id=14

            convBNLelu(256, 512, 3, 2, 1),
            ResidualBlock(512, 256),
            ResidualBlock(512, 256),
            ResidualBlock(512, 256),
            ResidualBlock(512, 256),
            ResidualBlock(512, 256),
            ResidualBlock(512, 256),
            ResidualBlock(512, 256),
            ResidualBlock(512, 256), # s16 id=23

            convBNLelu(512, 1024, 3, 2, 1),
            ResidualBlock(1024, 512),
            ResidualBlock(1024, 512),
            ResidualBlock(1024, 512),
            ResidualBlock(1024, 512), # s32 id=28

            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.Linear(1024,num_classes),
            nn.Softmax(-1)
        )

    def forward(self,x):
        return self.model(x)

        # x4 = self.model[:6](x)
        # x8 = self.model[6:15](x4)
        # x16 = self.model[15:24](x8)
        # x32 = self.model[24:29](x16)
        # return x4,x8,x16,x32

class Backbone_D53(nn.Module):
    def __init__(self,model_name="resnet18",pretrained=False,
                 freeze_at=["res1","res2","res3","res4","res5"]):
        super().__init__()
        _model = Darknet53(pretrained=pretrained)
        self.backbone =nn.ModuleDict(OrderedDict(
            [("res1",_model.model[:3]),
             ("res2",_model.model[3:6]),
             ("res3",_model.model[6:15]),
             ("res4",_model.model[15:24]),
             ("res5",_model.model[24:29])]
        ))
        # 参数冻结
        for name in freeze_at:
            for parameter in self.backbone[name].parameters():
                parameter.requires_grad_(False)

        # 统计所有可更新梯度的变量
        print("只有以下变量做梯度更新:")
        for name, parameter in self.backbone.named_parameters():
            if parameter.requires_grad:
                print("name:", name)

    def forward(self, x):
        out = {}
        x = self.backbone.res1(x)
        x = self.backbone.res2(x)
        out["res2"] = x
        x = self.backbone.res3(x)
        out["res3"] = x
        x = self.backbone.res4(x)
        out["res4"] = x
        x = self.backbone.res5(x)
        out["res5"] = x

        return out

class YoloV3Net(nn.Module):
    def __init__(self,num_classes=80,num_anchors=3,model_name="resnet18",pretrained=False,
                 freeze_at=["res1","res2","res3","res4","res5"]):
        super().__init__()
        out_filler = num_anchors*(5+num_classes)
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.backbone = Backbone_D53(model_name,pretrained,freeze_at)

        self.s32=nn.Sequential(
            convBNLelu(1024,512,1,1,0),
            convBNLelu(512,1024,3,1,1),
            convBNLelu(1024, 512, 1, 1, 0),
            convBNLelu(512, 1024, 3, 1, 1),
            convBNLelu(1024, 512, 1, 1, 0), # 4
            convBNLelu(512, 1024, 3, 1, 1),
            convBNLelu(1024, out_filler, 1, 1, 0),
        )

        self.s16=nn.Sequential(
            convBNLelu(512, 256, 1, 1, 0),
            nn.Upsample(scale_factor=2),

            convBNLelu(768, 256, 1, 1, 0),
            convBNLelu(256, 512, 3, 1, 1),
            convBNLelu(512, 256, 1, 1, 0),
            convBNLelu(256, 512, 3, 1, 1),
            convBNLelu(512, 256, 1, 1, 0), # 6
            convBNLelu(256, 512, 3, 1, 1),
            convBNLelu(512, out_filler, 1, 1, 0),
        )

        self.s8 = nn.Sequential(
            convBNLelu(256, 128, 1, 1, 0),
            nn.Upsample(scale_factor=2),

            convBNLelu(384, 128, 1, 1, 0),
            convBNLelu(128, 256, 3, 1, 1),
            convBNLelu(256, 128, 1, 1, 0),
            convBNLelu(128, 256, 3, 1, 1),
            convBNLelu(256, 128, 1, 1, 0),
            convBNLelu(128, 256, 3, 1, 1),
            convBNLelu(256, out_filler, 1, 1, 0),
        )


    def forward(self,x):
        out = self.backbone(x)

        x32_1=self.s32[:5](out["res5"])
        x32 = self.s32[5:](x32_1)

        x16_1 = self.s16[2:7](torch.cat((self.s16[:2](x32_1),out["res4"]),1))
        x16 = self.s16[7:](x16_1)

        x8 = self.s8[2:](torch.cat((self.s8[:2](x16_1),out["res3"]),1))

        x8 = x8.permute(0, 2, 3, 1)  # (-1,7,7,30)
        bs, h, w, c = x8.shape
        x8 = x8.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

        x16 = x16.permute(0, 2, 3, 1)  # (-1,7,7,30)
        bs, h, w, c = x16.shape
        x16 = x16.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

        x32 = x32.permute(0, 2, 3, 1)  # (-1,7,7,30)
        bs, h, w, c = x32.shape
        x32 = x32.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

        return x8,x16,x32

class YoloV3Net_spp(nn.Module):
    def __init__(self,num_classes=80,num_anchors=3,model_name="resnet18",pretrained=False,
                 freeze_at=["res1","res2","res3","res4","res5"],spp_branch=["s8","s16","s32"]):
        super().__init__()
        out_filler = num_anchors*(5+num_classes)
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.spp_branch = spp_branch
        self.backbone = Backbone_D53(model_name,pretrained,freeze_at)

        if len(spp_branch)>0:
            self.spp = nn.ModuleDict()
            for s in spp_branch:
                self.spp[s]=SPPNet()
                # self.spp[s]=SPPNetV2()

        self.s32=nn.Sequential(
            convBNLelu(1024,512,1,1,0),
            convBNLelu(512,1024,3,1,1),
            convBNLelu(1024, 512, 1, 1, 0),
            convBNLelu(512, 1024, 3, 1, 1),
            convBNLelu(1024, 512, 1, 1, 0), # 4
            convBNLelu(512, 1024, 3, 1, 1),
            convBNLelu(1024, out_filler, 1, 1, 0),
        )

        self.s16=nn.Sequential(
            convBNLelu(512, 256, 1, 1, 0),
            nn.Upsample(scale_factor=2),

            convBNLelu(768, 256, 1, 1, 0),
            convBNLelu(256, 512, 3, 1, 1),
            convBNLelu(512, 256, 1, 1, 0),
            convBNLelu(256, 512, 3, 1, 1),
            convBNLelu(512, 256, 1, 1, 0), # 6
            convBNLelu(256, 512, 3, 1, 1),
            convBNLelu(512, out_filler, 1, 1, 0),
        )

        self.s8 = nn.Sequential(
            convBNLelu(256, 128, 1, 1, 0),
            nn.Upsample(scale_factor=2),

            convBNLelu(384, 128, 1, 1, 0),
            convBNLelu(128, 256, 3, 1, 1),
            convBNLelu(256, 128, 1, 1, 0),
            convBNLelu(128, 256, 3, 1, 1),
            convBNLelu(256, 128, 1, 1, 0),
            convBNLelu(128, 256, 3, 1, 1),
            convBNLelu(256, out_filler, 1, 1, 0),
        )


    def forward(self,x):
        out = self.backbone(x)

        if len(self.spp_branch)>0:
            for s in self.spp_branch:
                if s=="s8":
                    out["res3"] = self.spp[s](out["res3"])
                elif s=="s16":
                    out["res4"] = self.spp[s](out["res4"])
                elif s=="s32":
                    out["res5"] = self.spp[s](out["res5"])

        x32_1=self.s32[:5](out["res5"])
        x32 = self.s32[5:](x32_1)

        x16_1 = self.s16[2:7](torch.cat((self.s16[:2](x32_1),out["res4"]),1))
        x16 = self.s16[7:](x16_1)

        x8 = self.s8[2:](torch.cat((self.s8[:2](x16_1),out["res3"]),1))

        x8 = x8.permute(0, 2, 3, 1)  # (-1,7,7,30)
        bs, h, w, c = x8.shape
        x8 = x8.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

        x16 = x16.permute(0, 2, 3, 1)  # (-1,7,7,30)
        bs, h, w, c = x16.shape
        x16 = x16.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

        x32 = x32.permute(0, 2, 3, 1)  # (-1,7,7,30)
        bs, h, w, c = x32.shape
        x32 = x32.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

        return x8,x16,x32

if __name__=="__main__":
    # m = Darknet53()
    # print(m)
    # print(m.state_dict().keys())
    # pred = m(torch.rand([1,3,256,256]))
    # print(pred.shape)

    # state_dict = torch.load("./weights/yolov3.pt")
    # print(state_dict["model"].keys())

    # m = Backbone_D53()
    m = ASPP(1024)
    print(m)
    pred = m(torch.rand([1, 1024, 13, 13]))
    print()