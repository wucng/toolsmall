try:
    from .yolo.models.yolo import Model
    from .yolo.models.common import *
except:
    from yolo.models.yolo import Model
    from yolo.models.common import *

import torch
from torch import nn
from collections import OrderedDict

__all__ =['YoloV3Net','YoloV3Head','YoloV5Net','YoloV5Head']

# m = Model("./yolo/models/yolov3.yaml",nc=80)
# print(m)
# exit(0)
# x = torch.rand([1,3,224,224])
# pred = m(x) # s=8,16,32，未加 sigmoid，【1,3,28,28,85】，【1,3,14,14,85】，【1,3,7,7,85】
# print()

class YoloV3Net(nn.Module):
    """
    - https://github.com/ultralytics/yolov3
    yolov3.yaml,yolov3-spp.yaml,(yolov3-tiny.yaml网络结构不一样 不能套用)
    """
    def __init__(self,cfg='yolov3.yaml', ch=3, nc=None,pretrained = False,weights="yolov3.pt",freeze_at=5,
                 num_anchors=3,num_classes=80,transform=0,only_features=False,only_head=False
                 ):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.transform = transform
        self.only_features = only_features
        self.only_head = only_head

        _m = Model(cfg,ch=ch,nc=nc)
        # print(_m)
        if pretrained:
            ckpt = torch.load(weights, map_location="cpu")
            # model.load_state_dict(ckpt['model'])
            ckpt["model"] = {k: v for k, v in ckpt["model"].items() if _m.state_dict()[k].numel() == v.numel()}
            _m.load_state_dict(ckpt["model"], strict=False)
            print("load weights... successful")

        self.backbone = nn.Sequential(
            OrderedDict([
                ("p1",_m.model[:3]),
                ("p2",_m.model[3:5]),
                ("p3",_m.model[5:7]),
                ("p4",_m.model[7:9]),
                ("p5",_m.model[9:11]),
            ]))

        if freeze_at > 0:
            for parameter in self.backbone[:freeze_at].parameters():
                parameter.requires_grad_(False)

        self.head = nn.Sequential(
            OrderedDict([
                ("p5",_m.model[11:16]),
                ("p4",_m.model[16:23]),
                ("p3",_m.model[23:28]),
            ]))

        self.detect = nn.ModuleList([
            nn.Conv2d(256,num_anchors*(5+num_classes),1),
            nn.Conv2d(512,num_anchors*(5+num_classes),1),
            nn.Conv2d(1024,num_anchors*(5+num_classes),1),
        ])

    def forward(self,x):
        x1 = self.backbone.p1(x)
        x2 = self.backbone.p2(x1)
        x3 = self.backbone.p3(x2)
        x4 = self.backbone.p4(x3)
        x5 = self.backbone.p5(x4)

        if self.only_features:
            return x2,x3,x4,x5

        _out5 = self.head.p5[:4](x5)
        out5 = self.head.p5[4](_out5)

        _out4=torch.cat((self.head.p4[:2](_out5),x4),1)
        _out4 = self.head.p4[3:-1](_out4)
        out4 = self.head.p4[-1](_out4)

        _out3 = torch.cat((self.head.p3[:2](_out4),x3),1)
        out3 = self.head.p3[3:](_out3)

        if self.only_head:
            return out3,out4,out5

        out3 = self.detect[0](out3)
        out4 = self.detect[1](out4)
        out5 = self.detect[2](out5)


        if self.transform==1:
            bs, c, h, w = out3.shape
            out3 = out3.contiguous().view(-1, self.num_anchors, self.num_classes + 5, h, w).permute(0, 1, 3, 4, 2)

            bs, c, h, w = out4.shape
            out4 = out4.contiguous().view(-1, self.num_anchors, self.num_classes + 5, h, w).permute(0, 1, 3, 4, 2)

            bs,c,h,w = out5.shape
            out5 = out5.contiguous().view(-1,self.num_anchors,self.num_classes+5,h,w).permute(0,1,3,4,2)
        elif self.transform==2:
            bs, c, h, w = out3.shape
            out3 = out3.permute(0, 2, 3, 1).contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

            bs, c, h, w = out4.shape
            out4 = out4.permute(0, 2, 3, 1).contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

            bs, c, h, w = out5.shape
            out5 = out5.permute(0, 2, 3, 1).contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

        return out3,out4,out5

class YoloV3Head(nn.Module):
    """FPN"""
    def __init__(self,in_channels=[256,512,1024],use_spp=True):
        super().__init__()
        self.p5 = nn.Sequential(
            Bottleneck(in_channels[-1], 1024, False),
            SPP(1024,512) if use_spp else Conv(1024, 512, 1, 1),
            Conv(512,1024,3,1),
            Conv(1024,512,1,1),
            Conv(512,1024,1,1))

        self.p4 = nn.Sequential(
            Conv(512,256,1,1),
            nn.Upsample(None,2,'nearest'),
            nn.Identity(),
            Bottleneck(768,512,False),
            Bottleneck(512,512,False),
            Conv(512, 256, 1, 1),
            Conv(256, 512, 3, 1),
        )

        self.p3 = nn.Sequential(
            Conv(256, 128, 1, 1),
            nn.Upsample(None, 2, 'nearest'),
            nn.Identity(),
            Bottleneck(384, 256, False),
            nn.Sequential(Bottleneck(256, 256, False),
            Bottleneck(256, 256, False))
        )


    def forward(self,x):
        """
        x[0] : s=8
        x[1] : s=16
        x[2] : s=32
        :param x:
        :return:
        """
        x3, x4, x5 = x
        _out5 = self.p5[:4](x5)
        out5 = self.p5[4](_out5)

        _out4 = torch.cat((self.p4[:2](_out5), x4), 1)
        _out4 = self.p4[3:-1](_out4)
        out4 = self.p4[-1](_out4)

        _out3 = torch.cat((self.p3[:2](_out4), x3), 1)
        out3 = self.p3[3:](_out3)

        return out3, out4, out5

class YoloV3HeadV2(nn.Module):
    """FPN"""
    def __init__(self,in_chs=[256,512,1024],use_spp=True):
        super().__init__()
        self.p5 = nn.Sequential(
            Bottleneck(in_chs[2], in_chs[2], False),
            SPP(in_chs[2],in_chs[1]) if use_spp else Conv(in_chs[2],in_chs[1], 1, 1),
            Conv(in_chs[1],in_chs[2],3,1),
            Conv(in_chs[2],in_chs[1],1,1),
            Conv(in_chs[1],in_chs[2],1,1))

        self.p4 = nn.Sequential(
            Conv(in_chs[1],in_chs[0],1,1),
            nn.Upsample(None,2,'nearest'),
            nn.Identity(),
            Bottleneck(in_chs[1]+in_chs[0],in_chs[1],False),
            Bottleneck(in_chs[1],in_chs[1],False),
            Conv(in_chs[1], in_chs[0], 1, 1),
            Conv(in_chs[0], in_chs[1], 3, 1),
        )

        self.p3 = nn.Sequential(
            Conv(in_chs[0], in_chs[0]//2, 1, 1),
            nn.Upsample(None, 2, 'nearest'),
            nn.Identity(),
            Bottleneck(in_chs[0]+in_chs[0]//2, in_chs[0], False),
            nn.Sequential(Bottleneck(in_chs[0], in_chs[0], False),
            Bottleneck(in_chs[0], in_chs[0], False))
        )


    def forward(self,x):
        """
        x[0] : s=8
        x[1] : s=16
        x[2] : s=32
        :param x:
        :return:
        """
        x3, x4, x5 = x
        _out5 = self.p5[:4](x5)
        out5 = self.p5[4](_out5)

        _out4 = torch.cat((self.p4[:2](_out5), x4), 1)
        _out4 = self.p4[3:-1](_out4)
        out4 = self.p4[-1](_out4)

        _out3 = torch.cat((self.p3[:2](_out4), x3), 1)
        out3 = self.p3[3:](_out3)

        return out3, out4, out5


class YoloV5Net(nn.Module):
    """
    - https://github.com/ultralytics/yolov5
    yolov5m.yaml [192,384,768]
    yolov5s.yaml [128,256,512]
    yolov5l.yaml [256,512,1024]
    yolov5x.yaml [320,640,1288]
    """
    def __init__(self,cfg='yolov5s.yaml', ch=3, nc=None,pretrained = False,weights="yolov5s.pt",freeze_at=5,
                 num_anchors=3,num_classes=80,transform=0,in_channels=[128,256,512],only_features=False,only_head=False):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.transform = transform
        self.only_features = only_features
        self.only_head = only_head

        _m = Model(cfg,ch=ch,nc=nc)
        # print(_m)
        # exit(0)
        if pretrained:
            ckpt = torch.load(weights, map_location="cpu")
            # model.load_state_dict(ckpt['model'])
            ckpt["model"] = {k: v for k, v in ckpt["model"].items() if _m.state_dict()[k].numel() == v.numel()}
            _m.load_state_dict(ckpt["model"], strict=False)
            print("load weights... successful")

        self.backbone = nn.Sequential(
            OrderedDict([
                ("p1",_m.model[0:1]),
                ("p2",_m.model[1:3]),
                ("p3",_m.model[3:5]),
                ("p4",_m.model[5:7]),
                ("p5",_m.model[7:10]),
            ]))

        if freeze_at > 0:
            for parameter in self.backbone[:freeze_at].parameters():
                parameter.requires_grad_(False)

        self.head = _m.model[10:-1]

        self.detect = nn.ModuleList([
            nn.Conv2d(in_channels[0],num_anchors*(5+num_classes),1),
            nn.Conv2d(in_channels[1],num_anchors*(5+num_classes),1),
            nn.Conv2d(in_channels[2],num_anchors*(5+num_classes),1),
        ])

    def forward(self,x):
        x1 = self.backbone.p1(x)
        x2 = self.backbone.p2(x1)
        x3 = self.backbone.p3(x2)
        x4 = self.backbone.p4(x3) # 6
        x5 = self.backbone.p5(x4)

        if self.only_features:
            return x2,x3,x4,x5

        _x5 = self.head[0](x5) # 10
        _x4 = torch.cat((self.head[1](_x5),x4),1)
        _x4 = self.head[3](_x4) # 13

        _x3 = self.head[4](_x4) # 14
        x3 = torch.cat((self.head[5](_x3),x3),1)
        x3 = self.head[7](x3) # 17

        _x4 = self.head[8](x3)
        _x4 = torch.cat((_x4,_x3),1)
        x4 = self.head[10](_x4)

        _x5 = torch.cat((self.head[11](x4),_x5),1)
        x5 = self.head[-1](_x5)

        if self.only_head:
            return x3,x4,x5

        out3 = self.detect[0](x3)
        out4 = self.detect[1](x4)
        out5 = self.detect[2](x5)

        if self.transform == 1:
            bs, c, h, w = out3.shape
            out3 = out3.contiguous().view(-1, self.num_anchors, self.num_classes + 5, h, w).permute(0, 1, 3, 4, 2)

            bs, c, h, w = out4.shape
            out4 = out4.contiguous().view(-1, self.num_anchors, self.num_classes + 5, h, w).permute(0, 1, 3, 4, 2)

            bs, c, h, w = out5.shape
            out5 = out5.contiguous().view(-1, self.num_anchors, self.num_classes + 5, h, w).permute(0, 1, 3, 4, 2)
        elif self.transform == 2:
            bs, c, h, w = out3.shape
            out3 = out3.permute(0, 2, 3, 1).contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

            bs, c, h, w = out4.shape
            out4 = out4.permute(0, 2, 3, 1).contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

            bs, c, h, w = out5.shape
            out5 = out5.permute(0, 2, 3, 1).contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

        return out3, out4, out5

class YoloV5Head(nn.Module):
    """FPN"""
    def __init__(self,in_channels=[256,512,1024]):
        super().__init__()

        self.head = nn.Sequential(
            Conv(in_channels[-1],512,1,1),
            nn.Upsample(None,2,'nearest'),
            nn.Identity(),
            C3(1024,512,3,False),

            Conv(512, 256, 1, 1),
            nn.Upsample(None, 2, 'nearest'),
            nn.Identity(),
            C3(512, 256, 3, False),

            Conv(256, 256, 3, 2),
            nn.Identity(),
            C3(512, 512, 3, False),

            Conv(512, 512, 3, 2),
            nn.Identity(),
            C3(1024, 1024, 3, False),
        )

    def forward(self,x):
        """
        x[0] : s=8
        x[1] : s=16
        x[2] : s=32
        :param x:
        :return:
        """
        x3, x4, x5 = x

        _x5 = self.head[0](x5)  # 10
        _x4 = torch.cat((self.head[1](_x5), x4), 1)
        _x4 = self.head[3](_x4)  # 13

        _x3 = self.head[4](_x4)  # 14
        x3 = torch.cat((self.head[5](_x3), x3), 1)
        x3 = self.head[7](x3)  # 17

        _x4 = self.head[8](x3)
        _x4 = torch.cat((_x4, _x3), 1)
        x4 = self.head[10](_x4)

        _x5 = torch.cat((self.head[11](x4), _x5), 1)
        x5 = self.head[-1](_x5)

        return x3, x4, x5


class YoloV5HeadV2(nn.Module):
    """FPN"""
    def __init__(self,in_chs=[256,512,1024]):
        super().__init__()

        self.spphead = nn.Sequential(
            SPP(in_chs[2],in_chs[2]),
            C3(in_chs[2],in_chs[2],3,False)
        )

        self.head = nn.Sequential(
            Conv(in_chs[2],in_chs[1],1,1),
            nn.Upsample(None,2,'nearest'),
            nn.Identity(),
            C3(in_chs[2],in_chs[1],3,False),

            Conv(in_chs[1], in_chs[0], 1, 1),
            nn.Upsample(None, 2, 'nearest'),
            nn.Identity(),
            C3(in_chs[1], in_chs[0], 3, False),

            Conv(in_chs[0], in_chs[0], 3, 2),
            nn.Identity(),
            C3(in_chs[1], in_chs[1], 3, False),

            Conv(in_chs[1], in_chs[1], 3, 2),
            nn.Identity(),
            C3(in_chs[2], in_chs[2], 3, False),
        )

    def forward(self,x):
        """
        x[0] : s=8
        x[1] : s=16
        x[2] : s=32
        :param x:
        :return:
        """
        x3, x4, x5 = x

        x5 = self.spphead(x5)
        
        _x5 = self.head[0](x5)  # 10
        _x4 = torch.cat((self.head[1](_x5), x4), 1)
        _x4 = self.head[3](_x4)  # 13

        _x3 = self.head[4](_x4)  # 14
        x3 = torch.cat((self.head[5](_x3), x3), 1)
        x3 = self.head[7](x3)  # 17

        _x4 = self.head[8](x3)
        _x4 = torch.cat((_x4, _x3), 1)
        x4 = self.head[10](_x4)

        _x5 = torch.cat((self.head[11](x4), _x5), 1)
        x5 = self.head[-1](_x5)

        return x3, x4, x5

if __name__ == "__main__":
    # m = YoloV3Net("./yolo/models/yolov3.yaml",nc=80,pretrained=True,weights="yolov3.pth")
    # m = YoloV5Net("./yolo/models/yolov5s.yaml",nc=80,pretrained=True,weights="yolov5s.pth")
    # m = YoloV5Net("./yolo/models/yolov5s.yaml",nc=80)
    # x = torch.rand([1, 3, 224, 224])
    # pred = m(x)
    # print()
    x=[torch.rand([1,128,28,28]),torch.rand([1,256,14,14]),torch.rand([1,512,7,7])]
    # head = YoloV3HeadV2([128,256,512],use_spp=False)
    head = YoloV5HeadV2([128,256,512])
    h=head(x)
    print()