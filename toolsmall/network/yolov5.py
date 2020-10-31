"""
yolov5 -v3.0

https://zhuanlan.zhihu.com/p/172121380
https://zhuanlan.zhihu.com/p/143747206
https://blog.csdn.net/Q1u1NG/article/details/107511465

https://github.com/ultralytics/yolov5
"""

import torch
from torch import nn
from torch.nn import functional as F
import os

from toolsmall.network.ultralytics.yolov5.models.common import *
from toolsmall.network.ultralytics.yolov5.utils.torch_utils import initialize_weights


class Detect(nn.Module):
    def __init__(self,num_classes=80,num_anchors=3,in_channels=[128,256,512]):
        super().__init__()
        self.m = nn.ModuleList()
        for in_channel in in_channels:
            self.m.append(nn.Conv2d(in_channel,num_anchors*(5+num_classes),1,1,0))

class Yolov5s(nn.Module):
    def __init__(self,num_classes=80,num_anchors=3,transform=True,pretrained=False,pretrained_weights="yolov5s.pt"):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.transform = transform

        self.model = nn.Sequential(
            Focus(3,32,3,1,1), # stride = 2
            Conv(32,64,3,2,1), # stride = 4
            BottleneckCSP(64,64,1),
            Conv(64,128,3,2,1), # stride = 8
            BottleneckCSP(128,128,3),

            Conv(128,256,3,2,1), # stride = 16
            BottleneckCSP(256,256,3),

            Conv(256,512,3,2,1), # stride = 32
            SPP(512,512),
            BottleneckCSP(512,512,1),
            Conv(512,256,1,1,0),

            nn.Upsample(scale_factor=2.0,mode="nearest"),
            Concat(),
            BottleneckCSP(512,256,1),
            Conv(256, 128, 1, 1, 0),

            nn.Upsample(scale_factor=2.0, mode="nearest"),
            Concat(),
            BottleneckCSP(256, 128, 1), # head

            Conv(128, 128, 3, 2, 1),
            Concat(),
            BottleneckCSP(256, 256, 1),  # head

            Conv(256, 256, 3, 2, 1),
            Concat(),
            BottleneckCSP(512, 512, 1),  # head

            Detect(num_classes,num_anchors,[128,256,512])

        )


        # 加载预训练权重
        if pretrained and os.path.exists(pretrained_weights):
            state_dict = torch.load(pretrained_weights)["model"].state_dict()
            _state_dict = {}
            i=0
            for k, v in self.model.state_dict().items():
                if "model."+k in state_dict and "model.24" not in "model."+k:
                    _state_dict[k] = state_dict["model."+k]
                else:
                    _state_dict[k] = v
                    i+=1

            self.model.load_state_dict(_state_dict)
            print("load weights success!!!! %d weights not load"%i)
        else:
            initialize_weights(self)


    def forward(self,x):
        x8 = self.model[:5](x)
        x16 = self.model[5:7](x8)
        x32 = self.model[7:11](x16)

        _x16 = self.model[12]((self.model[11](x32),x16))

        _x16 = self.model[13:15](_x16)
        head_s8 = self.model[17](self.model[16]((self.model[15](_x16),x8)))

        head_s16 = self.model[20](self.model[19]((self.model[18](head_s8),_x16)))

        head_s32 = self.model[23](self.model[22]((self.model[21](head_s16),x32)))

        head_s8 = self.model[24].m[0](head_s8)
        head_s16 = self.model[24].m[1](head_s16)
        head_s32 = self.model[24].m[2](head_s32)

        if self.transform:
            head_s8 = head_s8.permute(0, 2, 3, 1)  # (-1,7,7,30)
            bs, h, w, c = head_s8.shape
            head_s8 = head_s8.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

            head_s16 = head_s16.permute(0, 2, 3, 1)  # (-1,7,7,30)
            bs, h, w, c = head_s16.shape
            head_s16 = head_s16.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

            head_s32 = head_s32.permute(0, 2, 3, 1)  # (-1,7,7,30)
            bs, h, w, c = head_s32.shape
            head_s32 = head_s32.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

        return head_s8,head_s16,head_s32


class Yolov5m(nn.Module):
    def __init__(self,num_classes=80,num_anchors=3,transform=True,pretrained=False,pretrained_weights="yolov5m.pt"):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.transform = transform

        self.model = nn.Sequential(
            Focus(3,48,3,1,1), # stride = 2
            Conv(48,96,3,2,1), # stride = 4
            BottleneckCSP(96,96,2),
            Conv(96,192,3,2,1), # stride = 8
            BottleneckCSP(192,192,6),

            Conv(192,384,3,2,1), # stride = 16
            BottleneckCSP(384,384,6),

            Conv(384,768,3,2,1), # stride = 32
            SPP(768,768),
            BottleneckCSP(768,768,2),
            Conv(768,384,1,1,0),

            nn.Upsample(scale_factor=2.0,mode="nearest"),
            Concat(),
            BottleneckCSP(768,384,2),
            Conv(384, 192, 1, 1, 0),

            nn.Upsample(scale_factor=2.0, mode="nearest"),
            Concat(),
            BottleneckCSP(384, 192, 2), # head

            Conv(192, 192, 3, 2, 1),
            Concat(),
            BottleneckCSP(384, 384, 2),  # head

            Conv(384, 384, 3, 2, 1),
            Concat(),
            BottleneckCSP(768, 768, 2),  # head

            Detect(num_classes,num_anchors,[192,384,768])

        )

        # 加载预训练权重
        if pretrained and os.path.exists(pretrained_weights):
            state_dict = torch.load(pretrained_weights)["model"].state_dict()
            _state_dict = {}
            i = 0
            for k, v in self.model.state_dict().items():
                if "model." + k in state_dict and "model.24" not in "model." + k:
                    _state_dict[k] = state_dict["model." + k]
                else:
                    _state_dict[k] = v
                    i += 1

            self.model.load_state_dict(_state_dict)
            print("load weights success!!!! %d weights not load" % i)
        else:
            initialize_weights(self)

    def forward(self,x):
        x8 = self.model[:5](x)
        x16 = self.model[5:7](x8)
        x32 = self.model[7:11](x16)

        _x16 = self.model[12]((self.model[11](x32),x16))

        _x16 = self.model[13:15](_x16)
        head_s8 = self.model[17](self.model[16]((self.model[15](_x16),x8)))

        head_s16 = self.model[20](self.model[19]((self.model[18](head_s8),_x16)))

        head_s32 = self.model[23](self.model[22]((self.model[21](head_s16),x32)))

        head_s8 = self.model[24].m[0](head_s8)
        head_s16 = self.model[24].m[1](head_s16)
        head_s32 = self.model[24].m[2](head_s32)

        if self.transform:
            head_s8 = head_s8.permute(0, 2, 3, 1)  # (-1,7,7,30)
            bs, h, w, c = head_s8.shape
            head_s8 = head_s8.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

            head_s16 = head_s16.permute(0, 2, 3, 1)  # (-1,7,7,30)
            bs, h, w, c = head_s16.shape
            head_s16 = head_s16.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

            head_s32 = head_s32.permute(0, 2, 3, 1)  # (-1,7,7,30)
            bs, h, w, c = head_s32.shape
            head_s32 = head_s32.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

        return head_s8,head_s16,head_s32


class Yolov5l(nn.Module):
    def __init__(self,num_classes=80,num_anchors=3,transform=True,pretrained=False,pretrained_weights="yolov5l.pt"):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.transform = transform

        self.model = nn.Sequential(
            Focus(3,64,3,1,1), # stride = 2
            Conv(64,128,3,2,1), # stride = 4
            BottleneckCSP(128,128,3),
            Conv(128,256,3,2,1), # stride = 8
            BottleneckCSP(256,256,9),

            Conv(256,512,3,2,1), # stride = 16
            BottleneckCSP(512,512,9),

            Conv(512,1024,3,2,1), # stride = 32
            SPP(1024,1024),
            BottleneckCSP(1024,1024,3),
            Conv(1024,512,1,1,0),

            nn.Upsample(scale_factor=2.0,mode="nearest"),
            Concat(),
            BottleneckCSP(1024,512,3),
            Conv(512, 256, 1, 1, 0),

            nn.Upsample(scale_factor=2.0, mode="nearest"),
            Concat(),
            BottleneckCSP(512, 256, 3), # head 17

            Conv(256, 256, 3, 2, 1),
            Concat(),
            BottleneckCSP(512, 512, 3),  # head

            Conv(512, 512, 3, 2, 1),
            Concat(),
            BottleneckCSP(1024, 1024, 3),  # head

            Detect(num_classes,num_anchors,[256,512,1024])

        )

        # 加载预训练权重
        if pretrained and os.path.exists(pretrained_weights):
            state_dict = torch.load(pretrained_weights)["model"].state_dict()
            _state_dict = {}
            i = 0
            for k, v in self.model.state_dict().items():
                if "model." + k in state_dict and "model.24" not in "model." + k:
                    _state_dict[k] = state_dict["model." + k]
                else:
                    _state_dict[k] = v
                    i += 1

            self.model.load_state_dict(_state_dict)
            print("load weights success!!!! %d weights not load" % i)
        else:
            initialize_weights(self)

    def forward(self,x):
        x8 = self.model[:5](x)
        x16 = self.model[5:7](x8)
        x32 = self.model[7:11](x16)

        _x16 = self.model[12]((self.model[11](x32),x16))

        _x16 = self.model[13:15](_x16)
        head_s8 = self.model[17](self.model[16]((self.model[15](_x16),x8)))

        head_s16 = self.model[20](self.model[19]((self.model[18](head_s8),_x16)))

        head_s32 = self.model[23](self.model[22]((self.model[21](head_s16),x32)))

        head_s8 = self.model[24].m[0](head_s8)
        head_s16 = self.model[24].m[1](head_s16)
        head_s32 = self.model[24].m[2](head_s32)

        if self.transform:
            head_s8 = head_s8.permute(0, 2, 3, 1)  # (-1,7,7,30)
            bs, h, w, c = head_s8.shape
            head_s8 = head_s8.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

            head_s16 = head_s16.permute(0, 2, 3, 1)  # (-1,7,7,30)
            bs, h, w, c = head_s16.shape
            head_s16 = head_s16.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

            head_s32 = head_s32.permute(0, 2, 3, 1)  # (-1,7,7,30)
            bs, h, w, c = head_s32.shape
            head_s32 = head_s32.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

        return head_s8,head_s16,head_s32


class Yolov5x(nn.Module):
    def __init__(self,num_classes=80,num_anchors=3,transform=True,pretrained=False,pretrained_weights="yolov5x.pt"):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.transform = transform

        self.model = nn.Sequential(
            Focus(3,80,3,1,1), # stride = 2
            Conv(80,160,3,2,1), # stride = 4
            BottleneckCSP(160,160,4),
            Conv(160,320,3,2,1), # stride = 8
            BottleneckCSP(320,320,12),

            Conv(320,640,3,2,1), # stride = 16
            BottleneckCSP(640,640,12),

            Conv(640,1280,3,2,1), # stride = 32
            SPP(1280,1280),
            BottleneckCSP(1280,1280,4),
            Conv(1280,640,1,1,0),

            nn.Upsample(scale_factor=2.0,mode="nearest"),
            Concat(),
            BottleneckCSP(1280,640,4),
            Conv(640, 320, 1, 1, 0),

            nn.Upsample(scale_factor=2.0, mode="nearest"),
            Concat(),
            BottleneckCSP(640, 320, 4), # head 17

            Conv(320, 320, 3, 2, 1),
            Concat(),
            BottleneckCSP(640, 640, 4),  # head

            Conv(640, 640, 3, 2, 1),
            Concat(),
            BottleneckCSP(1280, 1280, 4),  # head

            Detect(num_classes,num_anchors,[320,640,1280])

        )

        # 加载预训练权重
        if pretrained and os.path.exists(pretrained_weights):
            state_dict = torch.load(pretrained_weights)["model"].state_dict()
            _state_dict = {}
            i = 0
            for k, v in self.model.state_dict().items():
                if "model." + k in state_dict and "model.24" not in "model." + k:
                    _state_dict[k] = state_dict["model." + k]
                else:
                    _state_dict[k] = v
                    i += 1

            self.model.load_state_dict(_state_dict)
            print("load weights success!!!! %d weights not load" % i)
        else:
            initialize_weights(self)

    def forward(self,x):
        x8 = self.model[:5](x)
        x16 = self.model[5:7](x8)
        x32 = self.model[7:11](x16)

        _x16 = self.model[12]((self.model[11](x32),x16))

        _x16 = self.model[13:15](_x16)
        head_s8 = self.model[17](self.model[16]((self.model[15](_x16),x8)))

        head_s16 = self.model[20](self.model[19]((self.model[18](head_s8),_x16)))

        head_s32 = self.model[23](self.model[22]((self.model[21](head_s16),x32)))

        head_s8 = self.model[24].m[0](head_s8)
        head_s16 = self.model[24].m[1](head_s16)
        head_s32 = self.model[24].m[2](head_s32)

        if self.transform:
            head_s8 = head_s8.permute(0, 2, 3, 1)  # (-1,7,7,30)
            bs, h, w, c = head_s8.shape
            head_s8 = head_s8.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

            head_s16 = head_s16.permute(0, 2, 3, 1)  # (-1,7,7,30)
            bs, h, w, c = head_s16.shape
            head_s16 = head_s16.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

            head_s32 = head_s32.permute(0, 2, 3, 1)  # (-1,7,7,30)
            bs, h, w, c = head_s32.shape
            head_s32 = head_s32.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

        return head_s8,head_s16,head_s32



if __name__=="__main__":
    x = torch.rand([1,3,608,608])
    path = "/media/wucong/225A6D42D4FA828F1/work/GitHub/toolkit/toolkit/papers/detection/YOLO/YOLOV3/yolov3_last/cfg/"
    path += "yolov5x.pt"
    network = Yolov5x(num_classes=20,pretrained=True,pretrained_weights=path)

