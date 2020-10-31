"""
yolov3，v4，yolov5 -v3.0

https://zhuanlan.zhihu.com/p/172121380
https://zhuanlan.zhihu.com/p/143747206
https://blog.csdn.net/Q1u1NG/article/details/107511465

https://github.com/ultralytics/yolov3
https://github.com/ultralytics/yolov5
https://github.com/AlexeyAB/darknet

pytorch 中不要使用 a+=b这种操作 （如果需要计算梯度，会导致无法计算梯度值）
"""

import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import os
try:
    from toolsmall.network.ultralytics.yolov3.utils.layers import WeightedFeatureFusion,Mish
    from toolsmall.network.ultralytics.yolov5.models.common import Focus,Conv,BottleneckCSP,SPP,Concat
    from toolsmall.network.ultralytics.yolov5.utils.torch_utils import initialize_weights
except:
    from .ultralytics.yolov3.utils.layers import WeightedFeatureFusion,Mish
    from .ultralytics.yolov5.models.common import Focus, Conv, BottleneckCSP, SPP, Concat
    from .ultralytics.yolov5.utils.torch_utils import initialize_weights

__all__=["Flatten","SPP","BottleneckCSP","Focus","BackBoneDarknet53","BackBoneCSPDarknet53",
         "Yolov3","Yolov3SPP","Yolov4","Yolov5s","Yolov5m","Yolov5l","Yolov5x","CBL","CBM"
         ]


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

def CBL(in_channels=3, out_channels=32,
               kernel_size=3, stride=1, padding=1,
               groups=1, bias=False,negative_slope=0.1):
    return nn.Sequential(OrderedDict([
            ('Conv2d',nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)),
            ('BatchNorm2d', nn.BatchNorm2d(out_channels, momentum=0.03, eps=1E-4)),
            ('activation', nn.LeakyReLU(negative_slope, inplace=True))
        ]))

def CBM(in_channels=3, out_channels=32,
               kernel_size=3, stride=1, padding=1,
               groups=1, bias=False):
    return nn.Sequential(OrderedDict([
            ('Conv2d',nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)),
            ('BatchNorm2d', nn.BatchNorm2d(out_channels, momentum=0.03, eps=1E-4)),
            # ('activation', nn.LeakyReLU(negative_slope, inplace=True))
            ('activation', Mish())
        ]))


class BackBoneDarknet53(nn.Module):
    """
    # convert darknet cfg/weights to pytorch model
    # from toolsmall.network.ultralytics.yolov3.models import convert
    $ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')"
    """
    def __init__(self,pretrained=False,pretrained_weights="yolov3.pt"):
        super().__init__()
        self.module_list = nn.ModuleList()
        self.module_list.append(CBL(3,32,3,1,1))

        self.module_list.append(CBL(32,64,3,2,1)) # stride = 2
        self.module_list.append(CBL(64,32,1,1,0))
        self.module_list.append(CBL(32,64,3,1,1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(64, 128, 3, 2, 1))  # stride = 4
        self.module_list.append(CBL(128, 64, 1, 1, 0))
        self.module_list.append(CBL(64, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(128, 64, 1, 1, 0))
        self.module_list.append(CBL(64, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(128, 256, 3, 2, 1)) # stride = 8
        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 512, 3, 2, 1))  # stride = 16
        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 1024, 3, 2, 1))  # stride = 32
        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        # ↑↑↑↑↑↑↑↑↑  backbone ↑↑↑↑↑↑↑↑↑↑↑

        # 加载预训练权重
        if pretrained and os.path.exists(pretrained_weights):
            state_dict = torch.load(pretrained_weights)["model"]#.state_dict()
            _state_dict = {}
            i = 0
            for k, v in self.module_list.state_dict().items():
                if "module_list."+k in state_dict:
                    _state_dict[k] = state_dict["module_list."+k]
                else:
                    _state_dict[k] = v
                    i+=1

            self.module_list.load_state_dict(_state_dict)
            print("load weights success!!!! %d weights not load"%i)
        else:
            initialize_weights(self)

        self.module_list = nn.Sequential(*self.module_list)

    def forward(self,x):
        x1 = self.module_list[:2](x)
        x1 = x1+self.module_list[2:4](x1)

        x4 = self.module_list[5](x1)
        x4 = x4+self.module_list[6:8](x4)
        x4 = x4+self.module_list[9:11](x4)

        x8 = self.module_list[12](x4)
        x8 = x8+self.module_list[13:15](x8)
        x8 = x8+self.module_list[16:18](x8)
        x8 = x8+self.module_list[19:21](x8)
        x8 = x8+self.module_list[22:24](x8)
        x8 = x8+self.module_list[25:27](x8)
        x8 = x8+ self.module_list[28:30](x8)
        x8 = x8+self.module_list[31:33](x8)
        x8 = x8+ self.module_list[34:36](x8)

        x16 = self.module_list[37](x8)
        x16 = x16 + self.module_list[38:40](x16)
        x16 = x16 + self.module_list[41:43](x16)
        x16 = x16 + self.module_list[44:46](x16)
        x16 = x16 + self.module_list[47:49](x16)
        x16 = x16 + self.module_list[50:52](x16)
        x16 = x16 + self.module_list[53:55](x16)
        x16 = x16 + self.module_list[56:58](x16)
        x16 = x16 + self.module_list[59:61](x16)

        x32 = self.module_list[62](x16)
        x32 = x32 + self.module_list[63:65](x32)
        x32 = x32 + self.module_list[66:68](x32)
        x32 = x32 + self.module_list[69:71](x32)
        x32 = x32 + self.module_list[72:74](x32)

        return x4,x8,x16,x32

class YOLOLayer(nn.Module):
    pass

class FeatureConcat(nn.Module):
    pass

class Yolov3(nn.Module):
    """
    # convert darknet cfg/weights to pytorch model
    # from toolsmall.network.ultralytics.yolov3.models import convert
    $ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')"
    """
    def __init__(self,num_classes=80,num_anchors=3,transform=True,pretrained=False,pretrained_weights="yolov3.pt"):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.transform = transform
        nums_out = self.num_anchors*(self.num_classes+5)

        self.module_list = nn.ModuleList()
        self.module_list.append(CBL(3,32,3,1,1))

        self.module_list.append(CBL(32,64,3,2,1)) # stride = 2
        self.module_list.append(CBL(64,32,1,1,0))
        self.module_list.append(CBL(32,64,3,1,1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(64, 128, 3, 2, 1))  # stride = 4
        self.module_list.append(CBL(128, 64, 1, 1, 0))
        self.module_list.append(CBL(64, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(128, 64, 1, 1, 0))
        self.module_list.append(CBL(64, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # idx = 11

        self.module_list.append(CBL(128, 256, 3, 2, 1)) # stride = 8
        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # idx = 36

        self.module_list.append(CBL(256, 512, 3, 2, 1))  # stride = 16
        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([])) # idx = 61

        self.module_list.append(CBL(512, 1024, 3, 2, 1))  # stride = 32
        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([])) # idx = 74

        # ↑↑↑↑↑↑↑↑↑  backbone ↑↑↑↑↑↑↑↑↑↑↑
        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1))
        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1))
        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1)) # 80
        self.module_list.append(nn.Sequential(OrderedDict([("Conv2d",nn.Conv2d(1024,nums_out,1,1,0))])))
        self.module_list.append(YOLOLayer())

        self.module_list.append(FeatureConcat()) # 83
        self.module_list.append(CBL(512,256,1,1,0)) # 84
        self.module_list.append(nn.Upsample(scale_factor=2.0,mode="nearest")) # idx = 85
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBL(768, 256, 1, 1, 0)) # 87
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(nn.Sequential(OrderedDict([("Conv2d", nn.Conv2d(512, nums_out, 1, 1, 0))])))
        self.module_list.append(YOLOLayer())

        self.module_list.append(FeatureConcat())
        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(nn.Upsample(scale_factor=2.0,mode="nearest")) # idx = 97
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBL(384, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(nn.Sequential(OrderedDict([("Conv2d", nn.Conv2d(256, nums_out, 1, 1, 0))])))
        self.module_list.append(YOLOLayer())


        # 加载预训练权重
        if pretrained and os.path.exists(pretrained_weights):
            state_dict = torch.load(pretrained_weights)["model"]#.state_dict()
            _state_dict = {}
            i = 0
            for k, v in self.module_list.state_dict().items():
                tk = "module_list."+k
                if tk in state_dict and "module_list.81" not in tk and \
                        "module_list.93" not in tk and "module_list.105" not in tk:
                    _state_dict[k] = state_dict[tk]
                else:
                    _state_dict[k] = v
                    i+=1

            self.module_list.load_state_dict(_state_dict)
            print("load weights success!!!! %d weights not load"%i)
        else:
            initialize_weights(self)

        self.module_list = nn.Sequential(*self.module_list)

    def forward(self,x):
        x1 = self.module_list[:2](x)
        x1 = x1+self.module_list[2:4](x1)

        x4 = self.module_list[5](x1)
        x4 = x4+self.module_list[6:8](x4)
        x4 = x4+self.module_list[9:11](x4)

        x8 = self.module_list[12](x4)
        x8 = x8+self.module_list[13:15](x8)
        x8 = x8+self.module_list[16:18](x8)
        x8 = x8+self.module_list[19:21](x8)
        x8 = x8+self.module_list[22:24](x8)
        x8 = x8+self.module_list[25:27](x8)
        x8 = x8+self.module_list[28:30](x8)
        x8 = x8+self.module_list[31:33](x8)
        x8 = x8+self.module_list[34:36](x8)

        x16 = self.module_list[37](x8)
        x16 = x16+self.module_list[38:40](x16)
        x16 = x16+self.module_list[41:43](x16)
        x16 = x16+self.module_list[44:46](x16)
        x16 = x16+self.module_list[47:49](x16)
        x16 = x16+self.module_list[50:52](x16)
        x16 = x16+self.module_list[53:55](x16)
        x16 = x16+self.module_list[56:58](x16)
        x16 = x16+self.module_list[59:61](x16)

        x32 = self.module_list[62](x16)
        x32 = x32+self.module_list[63:65](x32)
        x32 = x32+self.module_list[66:68](x32)
        x32 = x32+self.module_list[69:71](x32)
        x32 = x32+self.module_list[72:74](x32)

        # ↑↑↑↑↑↑↑↑↑  backbone ↑↑↑↑↑↑↑↑↑↑↑
        x32 = self.module_list[75:77](x32)
        x32 = self.module_list[77:79](x32)
        _x32 = self.module_list[79](x32)
        x32 = self.module_list[80](_x32)
        x32 = self.module_list[81](x32) # out

        _x16 = self.module_list[84:86](_x32)
        x16 = torch.cat((_x16,x16),1)
        x16 = self.module_list[87:89](x16)
        x16 = self.module_list[89:91](x16)
        _x16 = self.module_list[91](x16)
        x16 = self.module_list[92](_x16)
        x16 = self.module_list[93](x16)  # out

        _x8 = self.module_list[96:98](_x16)
        x8 = torch.cat((_x8,x8), 1)
        x8 = self.module_list[99:101](x8)
        x8 = self.module_list[101:103](x8)
        x8 = self.module_list[103:105](x8)
        x8 = self.module_list[105](x8)  # out

        if self.transform:
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


class Yolov3SPP(nn.Module):
    """
    # convert darknet cfg/weights to pytorch model
    # from toolsmall.network.ultralytics.yolov3.models import convert
    $ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')"
    """
    def __init__(self,num_classes=80,num_anchors=3,transform=True,pretrained=False,pretrained_weights="yolov3-spp.pt"):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.transform = transform
        nums_out = self.num_anchors*(self.num_classes+5)

        self.module_list = nn.ModuleList()
        self.module_list.append(CBL(3,32,3,1,1))

        self.module_list.append(CBL(32,64,3,2,1)) # stride = 2
        self.module_list.append(CBL(64,32,1,1,0))
        self.module_list.append(CBL(32,64,3,1,1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(64, 128, 3, 2, 1))  # stride = 4
        self.module_list.append(CBL(128, 64, 1, 1, 0))
        self.module_list.append(CBL(64, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(128, 64, 1, 1, 0))
        self.module_list.append(CBL(64, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # idx = 11

        self.module_list.append(CBL(128, 256, 3, 2, 1)) # stride = 8
        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # idx = 36

        self.module_list.append(CBL(256, 512, 3, 2, 1))  # stride = 16
        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([])) # idx = 61

        self.module_list.append(CBL(512, 1024, 3, 2, 1))  # stride = 32
        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([])) # idx = 74

        # ↑↑↑↑↑↑↑↑↑  backbone ↑↑↑↑↑↑↑↑↑↑↑
        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1))
        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        # spp
        self.module_list.append(nn.MaxPool2d(5,1,2)) # 78
        self.module_list.append(FeatureConcat())
        self.module_list.append(nn.MaxPool2d(9, 1, 4))  # 80
        self.module_list.append(FeatureConcat())
        self.module_list.append(nn.MaxPool2d(13, 1, 6))  # 82
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBL(2048,512,1,1,0))

        self.module_list.append(CBL(512, 1024, 3, 1, 1)) # 85
        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1)) # 87
        self.module_list.append(nn.Sequential(OrderedDict([("Conv2d",nn.Conv2d(1024,nums_out,1,1,0))])))
        self.module_list.append(YOLOLayer())

        self.module_list.append(FeatureConcat()) # 90
        self.module_list.append(CBL(512,256,1,1,0))
        self.module_list.append(nn.Upsample(scale_factor=2.0,mode="nearest")) # idx = 91
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBL(768, 256, 1, 1, 0)) # 93
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1)) # 98
        self.module_list.append(nn.Sequential(OrderedDict([("Conv2d", nn.Conv2d(512, nums_out, 1, 1, 0))])))
        self.module_list.append(YOLOLayer())

        self.module_list.append(FeatureConcat())
        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(nn.Upsample(scale_factor=2.0,mode="nearest")) # idx = 103
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBL(384, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(nn.Sequential(OrderedDict([("Conv2d", nn.Conv2d(256, nums_out, 1, 1, 0))])))
        self.module_list.append(YOLOLayer())


        # 加载预训练权重
        if pretrained and os.path.exists(pretrained_weights):
            state_dict = torch.load(pretrained_weights)["model"]#.state_dict()
            _state_dict = {}
            i = 0
            for k, v in self.module_list.state_dict().items():
                tk = "module_list."+k
                if tk in state_dict and "module_list.88" not in tk and \
                        "module_list.100" not in tk and "module_list.112" not in tk:
                    _state_dict[k] = state_dict[tk]
                else:
                    _state_dict[k] = v
                    i+=1

            self.module_list.load_state_dict(_state_dict)
            print("load weights success!!!! %d weights not load"%i)
        else:
            initialize_weights(self)

        self.module_list = nn.Sequential(*self.module_list)

    def forward(self,x):
        x1 = self.module_list[:2](x)
        x1 = x1+self.module_list[2:4](x1)

        x4 = self.module_list[5](x1)
        x4 = x4+self.module_list[6:8](x4)
        x4 = x4+self.module_list[9:11](x4)

        x8 = self.module_list[12](x4)
        x8 = x8+self.module_list[13:15](x8)
        x8 = x8+self.module_list[16:18](x8)
        x8 = x8+self.module_list[19:21](x8)
        x8 = x8+self.module_list[22:24](x8)
        x8 = x8+self.module_list[25:27](x8)
        x8 = x8+self.module_list[28:30](x8)
        x8 = x8+self.module_list[31:33](x8)
        x8 = x8+self.module_list[34:36](x8)

        x16 = self.module_list[37](x8)
        x16 = x16+self.module_list[38:40](x16)
        x16 = x16+self.module_list[41:43](x16)
        x16 = x16+self.module_list[44:46](x16)
        x16 = x16+self.module_list[47:49](x16)
        x16 = x16+self.module_list[50:52](x16)
        x16 = x16+self.module_list[53:55](x16)
        x16 = x16+self.module_list[56:58](x16)
        x16 = x16+self.module_list[59:61](x16)

        x32 = self.module_list[62](x16)
        x32 = x32+self.module_list[63:65](x32)
        x32 = x32+self.module_list[66:68](x32)
        x32 = x32+self.module_list[69:71](x32)
        x32 = x32+self.module_list[72:74](x32)

        # ↑↑↑↑↑↑↑↑↑  backbone ↑↑↑↑↑↑↑↑↑↑↑
        x32 = self.module_list[75:78](x32)
        # 加上 spp ------------------------------
        x32 = torch.cat((x32,self.module_list[78](x32),self.module_list[80](x32),self.module_list[82](x32)),1)
        x32 = self.module_list[84](x32)
        # ------------------------------
        x32 = self.module_list[85](x32)

        _x32 = self.module_list[86](x32)
        x32 = self.module_list[87](_x32)
        x32 = self.module_list[88](x32) # out

        _x16 = self.module_list[91:93](_x32)
        x16 = torch.cat((_x16,x16),1)
        x16 = self.module_list[94:96](x16)
        x16 = self.module_list[96:98](x16)
        _x16 = self.module_list[98](x16)
        x16 = self.module_list[99](_x16)
        x16 = self.module_list[100](x16)  # out

        _x8 = self.module_list[103:105](_x16)
        x8 = torch.cat((_x8,x8), 1)
        x8 = self.module_list[106:108](x8)
        x8 = self.module_list[108:110](x8)
        x8 = self.module_list[110:112](x8)
        x8 = self.module_list[112](x8)  # out

        if self.transform:
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


class BackBoneCSPDarknet53(nn.Module):
    """
    # convert darknet cfg/weights to pytorch model
    # from toolsmall.network.ultralytics.yolov3.models import convert
    $ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')"
    """
    def __init__(self,pretrained=False,pretrained_weights="yolov4.pt"):
        super().__init__()
        self.module_list = nn.ModuleList()
        self.module_list.append(CBM(3,32,3,1,1))

        # CSP
        self.module_list.append(CBM(32,64,3,2,1)) # stride = 2
        self.module_list.append(CBM(64, 64, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(64, 64, 1, 1, 0))

        self.module_list.append(CBM(64, 32, 1, 1, 0)) # 5
        self.module_list.append(CBM(32, 64, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBM(64, 64, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(128, 64, 1, 1, 0)) # 10

        # CSP
        self.module_list.append(CBM(64, 128, 3, 2, 1))  # stride = 4
        self.module_list.append(CBM(128, 64, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(128, 64, 1, 1, 0))

        self.module_list.append(CBM(64, 64, 1, 1, 0)) # 15
        self.module_list.append(CBM(64, 64, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([])) # 17
        self.module_list.append(CBM(64, 64, 1, 1, 0))
        self.module_list.append(CBM(64, 64, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 20

        self.module_list.append(CBM(64, 64, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(128, 128, 1, 1, 0))  # 23

        # CSP
        self.module_list.append(CBM(128,256,3,2,1)) # stride = 8
        self.module_list.append(CBM(256, 128, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(256, 128, 1, 1, 0))

        self.module_list.append(CBM(128, 128, 1, 1, 0))  # 28
        self.module_list.append(CBM(128, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 30
        self.module_list.append(CBM(128, 128, 1, 1, 0))  # 31
        self.module_list.append(CBM(128, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 33
        self.module_list.append(CBM(128, 128, 1, 1, 0))  # 34
        self.module_list.append(CBM(128, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 36
        self.module_list.append(CBM(128, 128, 1, 1, 0))  # 37
        self.module_list.append(CBM(128, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 39
        self.module_list.append(CBM(128, 128, 1, 1, 0))  # 40
        self.module_list.append(CBM(128, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 42
        self.module_list.append(CBM(128, 128, 1, 1, 0))  # 43
        self.module_list.append(CBM(128, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 45
        self.module_list.append(CBM(128, 128, 1, 1, 0))  # 46
        self.module_list.append(CBM(128, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 48
        self.module_list.append(CBM(128, 128, 1, 1, 0))  # 49
        self.module_list.append(CBM(128, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 51

        self.module_list.append(CBM(128, 128, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(256, 256, 1, 1, 0))  # 54

        # CSP
        self.module_list.append(CBM(256, 512, 3, 2, 1))  # stride = 16
        self.module_list.append(CBM(512, 256, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(512, 256, 1, 1, 0)) # 58

        self.module_list.append(CBM(256, 256, 1, 1, 0))  # 59
        self.module_list.append(CBM(256, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 61
        self.module_list.append(CBM(256, 256, 1, 1, 0))  # 62
        self.module_list.append(CBM(256, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 64
        self.module_list.append(CBM(256, 256, 1, 1, 0))  # 65
        self.module_list.append(CBM(256, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 67
        self.module_list.append(CBM(256, 256, 1, 1, 0))  # 68
        self.module_list.append(CBM(256, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 70
        self.module_list.append(CBM(256, 256, 1, 1, 0))  # 71
        self.module_list.append(CBM(256, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 73
        self.module_list.append(CBM(256, 256, 1, 1, 0))  # 74
        self.module_list.append(CBM(256, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 76
        self.module_list.append(CBM(256, 256, 1, 1, 0))  # 77
        self.module_list.append(CBM(256, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 79
        self.module_list.append(CBM(256, 256, 1, 1, 0))  # 80
        self.module_list.append(CBM(256, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 82

        self.module_list.append(CBM(256, 256, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(512, 512, 1, 1, 0))  # 85

        # CSP
        self.module_list.append(CBM(512,1024, 3, 2, 1))  # stride = 32
        self.module_list.append(CBM(1024, 512, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(1024, 512, 1, 1, 0))  # 89

        self.module_list.append(CBM(512, 512, 1, 1, 0))  # 90
        self.module_list.append(CBM(512, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 92
        self.module_list.append(CBM(512, 512, 1, 1, 0))  # 93
        self.module_list.append(CBM(512, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 95
        self.module_list.append(CBM(512, 512, 1, 1, 0))  # 96
        self.module_list.append(CBM(512, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 98
        self.module_list.append(CBM(512, 512, 1, 1, 0))  # 99
        self.module_list.append(CBM(512, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 101

        self.module_list.append(CBM(512, 512, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(1024, 1024, 1, 1, 0))  # 104



        # 加载预训练权重
        if pretrained and os.path.exists(pretrained_weights):
            state_dict = torch.load(pretrained_weights)["model"]#.state_dict()
            _state_dict = {}
            i = 0
            for k, v in self.module_list.state_dict().items():
                tk = "module_list."+k
                if tk in state_dict:
                    _state_dict[k] = state_dict[tk]
                else:
                    _state_dict[k] = v
                    i+=1

            self.module_list.load_state_dict(_state_dict)
            print("load weights success!!!! %d weights not load"%i)
        else:
            initialize_weights(self)

        self.module_list = nn.Sequential(*self.module_list)

    def forward(self,x):
        x1 = self.module_list[:2](x)
        x11 = self.module_list[2](x1)
        x12 = self.module_list[4](x1)
        x12 = x12+self.module_list[5:7](x12)
        x12 = self.module_list[8](x12)
        x1 = torch.cat((x12,x11),1)
        x1 = self.module_list[10](x1)

        x4 = self.module_list[11](x1)
        x41 = self.module_list[12](x4)
        x42 = self.module_list[14](x4)
        x42 = x42+self.module_list[15:17](x42)
        x42 = x42+self.module_list[18:20](x42)
        x42 = self.module_list[21](x42)
        x4 = torch.cat((x42,x41),1)
        x4 = self.module_list[23](x4)

        x8 = self.module_list[24](x4)
        x81 = self.module_list[25](x8)
        x82 = self.module_list[27](x8)
        x82 = x82+self.module_list[28:30](x82)
        x82 = x82+self.module_list[31:33](x82)
        x82 = x82+self.module_list[34:36](x82)
        x82 = x82+self.module_list[37:39](x82)
        x82 = x82+self.module_list[40:42](x82)
        x82 = x82+self.module_list[43:45](x82)
        x82 = x82+self.module_list[46:48](x82)
        x82 = x82+self.module_list[49:51](x82)
        x82 = self.module_list[52](x82)
        x8 = torch.cat((x82,x81),1)
        x8 = self.module_list[54](x8)

        x16 = self.module_list[55](x8)
        x16_1 = self.module_list[56](x16)
        x16_2 = self.module_list[58](x16)
        x16_2 = x16_2+self.module_list[59:61](x16_2)
        x16_2 = x16_2+self.module_list[62:64](x16_2)
        x16_2 = x16_2+self.module_list[65:67](x16_2)
        x16_2 = x16_2+self.module_list[68:70](x16_2)
        x16_2 = x16_2+self.module_list[71:73](x16_2)
        x16_2 = x16_2+self.module_list[74:76](x16_2)
        x16_2 = x16_2+self.module_list[77:79](x16_2)
        x16_2 = x16_2+self.module_list[80:82](x16_2)
        x16_2 = self.module_list[83](x16_2)
        x16 = torch.cat((x16_2,x16_1),1)
        x16 = self.module_list[85](x16)

        x32 = self.module_list[86](x16)
        x32_1 = self.module_list[87](x32)
        x32_2 = self.module_list[89](x32)
        x32_2 = x32_2+self.module_list[90:92](x32_2)
        x32_2 = x32_2+self.module_list[93:95](x32_2)
        x32_2 = x32_2+self.module_list[96:98](x32_2)
        x32_2 = x32_2+self.module_list[99:101](x32_2)
        x32_2 = self.module_list[102](x32_2)
        x32 = torch.cat((x32_2,x32_1),1)
        x32 = self.module_list[104](x32)


        return x4,x8,x16,x32


class Yolov4(nn.Module):
    """
    # convert darknet cfg/weights to pytorch model
    # from toolsmall.network.ultralytics.yolov3.models import convert
    $ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')"
    """
    def __init__(self,num_classes=80,num_anchors=3,transform=True,pretrained=False,pretrained_weights="yolov4.pt"):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.transform = transform
        nums_out = self.num_anchors*(self.num_classes+5)

        self.module_list = nn.ModuleList()
        self.module_list.append(CBM(3,32,3,1,1))

        # CSP
        self.module_list.append(CBM(32,64,3,2,1)) # stride = 2
        self.module_list.append(CBM(64, 64, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(64, 64, 1, 1, 0))

        self.module_list.append(CBM(64, 32, 1, 1, 0)) # 5
        self.module_list.append(CBM(32, 64, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))

        self.module_list.append(CBM(64, 64, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(128, 64, 1, 1, 0)) # 10

        # CSP
        self.module_list.append(CBM(64, 128, 3, 2, 1))  # stride = 4
        self.module_list.append(CBM(128, 64, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(128, 64, 1, 1, 0))

        self.module_list.append(CBM(64, 64, 1, 1, 0)) # 15
        self.module_list.append(CBM(64, 64, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([])) # 17
        self.module_list.append(CBM(64, 64, 1, 1, 0))
        self.module_list.append(CBM(64, 64, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 20

        self.module_list.append(CBM(64, 64, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(128, 128, 1, 1, 0))  # 23

        # CSP
        self.module_list.append(CBM(128,256,3,2,1)) # stride = 8
        self.module_list.append(CBM(256, 128, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(256, 128, 1, 1, 0))

        self.module_list.append(CBM(128, 128, 1, 1, 0))  # 28
        self.module_list.append(CBM(128, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 30
        self.module_list.append(CBM(128, 128, 1, 1, 0))  # 31
        self.module_list.append(CBM(128, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 33
        self.module_list.append(CBM(128, 128, 1, 1, 0))  # 34
        self.module_list.append(CBM(128, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 36
        self.module_list.append(CBM(128, 128, 1, 1, 0))  # 37
        self.module_list.append(CBM(128, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 39
        self.module_list.append(CBM(128, 128, 1, 1, 0))  # 40
        self.module_list.append(CBM(128, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 42
        self.module_list.append(CBM(128, 128, 1, 1, 0))  # 43
        self.module_list.append(CBM(128, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 45
        self.module_list.append(CBM(128, 128, 1, 1, 0))  # 46
        self.module_list.append(CBM(128, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 48
        self.module_list.append(CBM(128, 128, 1, 1, 0))  # 49
        self.module_list.append(CBM(128, 128, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 51

        self.module_list.append(CBM(128, 128, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(256, 256, 1, 1, 0))  # 54

        # CSP
        self.module_list.append(CBM(256, 512, 3, 2, 1))  # stride = 16
        self.module_list.append(CBM(512, 256, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(512, 256, 1, 1, 0)) # 58

        self.module_list.append(CBM(256, 256, 1, 1, 0))  # 59
        self.module_list.append(CBM(256, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 61
        self.module_list.append(CBM(256, 256, 1, 1, 0))  # 62
        self.module_list.append(CBM(256, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 64
        self.module_list.append(CBM(256, 256, 1, 1, 0))  # 65
        self.module_list.append(CBM(256, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 67
        self.module_list.append(CBM(256, 256, 1, 1, 0))  # 68
        self.module_list.append(CBM(256, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 70
        self.module_list.append(CBM(256, 256, 1, 1, 0))  # 71
        self.module_list.append(CBM(256, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 73
        self.module_list.append(CBM(256, 256, 1, 1, 0))  # 74
        self.module_list.append(CBM(256, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 76
        self.module_list.append(CBM(256, 256, 1, 1, 0))  # 77
        self.module_list.append(CBM(256, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 79
        self.module_list.append(CBM(256, 256, 1, 1, 0))  # 80
        self.module_list.append(CBM(256, 256, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 82

        self.module_list.append(CBM(256, 256, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(512, 512, 1, 1, 0))  # 85

        # CSP
        self.module_list.append(CBM(512,1024, 3, 2, 1))  # stride = 32
        self.module_list.append(CBM(1024, 512, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(1024, 512, 1, 1, 0))  # 89

        self.module_list.append(CBM(512, 512, 1, 1, 0))  # 90
        self.module_list.append(CBM(512, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 92
        self.module_list.append(CBM(512, 512, 1, 1, 0))  # 93
        self.module_list.append(CBM(512, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 95
        self.module_list.append(CBM(512, 512, 1, 1, 0))  # 96
        self.module_list.append(CBM(512, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 98
        self.module_list.append(CBM(512, 512, 1, 1, 0))  # 99
        self.module_list.append(CBM(512, 512, 3, 1, 1))
        self.module_list.append(WeightedFeatureFusion([]))  # 101

        self.module_list.append(CBM(512, 512, 1, 1, 0))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBM(1024, 1024, 1, 1, 0))  # 104

        # ↑↑↑↑↑↑↑↑↑  backbone ↑↑↑↑↑↑↑↑↑↑↑

        self.module_list.append(CBL(1024,512,1,1,0)) # 105
        self.module_list.append(CBL(512,1024,3,1,1)) # 106
        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        # spp
        self.module_list.append(nn.MaxPool2d(5, 1, 2))  # 108
        self.module_list.append(FeatureConcat())
        self.module_list.append(nn.MaxPool2d(9, 1, 4))  # 110
        self.module_list.append(FeatureConcat())
        self.module_list.append(nn.MaxPool2d(13, 1, 6))  # 112
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBL(2048, 512, 1, 1, 0)) # 114

        self.module_list.append(CBL(512, 1024, 3, 1, 1))  # 115
        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(nn.Upsample(scale_factor=2.0,mode="nearest")) # 118
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBL(512,256,1,1,0))
        self.module_list.append(FeatureConcat()) # 121
        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(CBL(512, 256, 1, 1, 0)) # 124
        self.module_list.append(CBL(256, 512, 3, 1, 1))
        self.module_list.append(CBL(512, 256, 1, 1, 0))  # 126
        self.module_list.append(CBL(256, 128, 1, 1, 0))  # 127
        self.module_list.append(nn.Upsample(scale_factor=2.0, mode="nearest"))  # 128
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBL(256, 128, 1, 1, 0)) # 130
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1)) # 133
        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1)) # 135
        self.module_list.append(CBL(256, 128, 1, 1, 0))
        self.module_list.append(CBL(128, 256, 3, 1, 1))
        self.module_list.append(nn.Sequential(OrderedDict([("Conv2d", nn.Conv2d(256, nums_out, 1, 1, 0))])))
        self.module_list.append(YOLOLayer())

        self.module_list.append(FeatureConcat()) # 140
        self.module_list.append(CBL(128, 256, 3, 2, 1))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1)) # 144
        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))  # 146
        self.module_list.append(CBL(512, 256, 1, 1, 0))
        self.module_list.append(CBL(256, 512, 3, 1, 1))  # 148
        self.module_list.append(nn.Sequential(OrderedDict([("Conv2d", nn.Conv2d(512, nums_out, 1, 1, 0))])))
        self.module_list.append(YOLOLayer())

        self.module_list.append(FeatureConcat())  # 151
        self.module_list.append(CBL(256, 512, 3, 2, 1))
        self.module_list.append(FeatureConcat())
        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1)) # 155
        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1))  # 157
        self.module_list.append(CBL(1024, 512, 1, 1, 0))
        self.module_list.append(CBL(512, 1024, 3, 1, 1))  # 159
        self.module_list.append(nn.Sequential(OrderedDict([("Conv2d", nn.Conv2d(1024, nums_out, 1, 1, 0))])))
        self.module_list.append(YOLOLayer())


        # 加载预训练权重
        if pretrained and os.path.exists(pretrained_weights):
            state_dict = torch.load(pretrained_weights)["model"]#.state_dict()
            _state_dict = {}
            i = 0
            for k, v in self.module_list.state_dict().items():
                tk = "module_list."+k
                if tk in state_dict and "module_list.138" not in tk and \
                        "module_list.149" not in tk and "module_list.160" not in tk:
                    _state_dict[k] = state_dict[tk]
                else:
                    _state_dict[k] = v
                    i+=1

            self.module_list.load_state_dict(_state_dict)
            print("load weights success!!!! %d weights not load"%i)
        else:
            initialize_weights(self)

        self.module_list = nn.Sequential(*self.module_list)

    def forward(self,x):
        x1 = self.module_list[:2](x)
        x11 = self.module_list[2](x1)
        x12 = self.module_list[4](x1)
        x12 = x12+self.module_list[5:7](x12)
        x12 = self.module_list[8](x12)
        x1 = torch.cat((x12,x11),1)
        x1 = self.module_list[10](x1)

        x4 = self.module_list[11](x1)
        x41 = self.module_list[12](x4)
        x42 = self.module_list[14](x4)
        x42 = x42+self.module_list[15:17](x42)
        x42 = x42+self.module_list[18:20](x42)
        x42 = self.module_list[21](x42)
        x4 = torch.cat((x42,x41),1)
        x4 = self.module_list[23](x4)

        x8 = self.module_list[24](x4)
        x81 = self.module_list[25](x8)
        x82 = self.module_list[27](x8)
        x82 = x82+self.module_list[28:30](x82)
        x82 = x82+self.module_list[31:33](x82)
        x82 = x82+self.module_list[34:36](x82)
        x82 = x82+self.module_list[37:39](x82)
        x82 = x82+self.module_list[40:42](x82)
        x82 = x82+self.module_list[43:45](x82)
        x82 = x82+self.module_list[46:48](x82)
        x82 = x82+self.module_list[49:51](x82)
        x82 = self.module_list[52](x82)
        x8 = torch.cat((x82,x81),1)
        x8 = self.module_list[54](x8)

        x16 = self.module_list[55](x8)
        x16_1 = self.module_list[56](x16)
        x16_2 = self.module_list[58](x16)
        x16_2 = x16_2+self.module_list[59:61](x16_2)
        x16_2 = x16_2+self.module_list[62:64](x16_2)
        x16_2 = x16_2+self.module_list[65:67](x16_2)
        x16_2 = x16_2+self.module_list[68:70](x16_2)
        x16_2 = x16_2+self.module_list[71:73](x16_2)
        x16_2 = x16_2+self.module_list[74:76](x16_2)
        x16_2 = x16_2+self.module_list[77:79](x16_2)
        x16_2 = x16_2+self.module_list[80:82](x16_2)
        x16_2 = self.module_list[83](x16_2)
        x16 = torch.cat((x16_2,x16_1),1)
        x16 = self.module_list[85](x16)

        x32 = self.module_list[86](x16)
        x32_1 = self.module_list[87](x32)
        x32_2 = self.module_list[89](x32)
        x32_2 = x32_2+self.module_list[90:92](x32_2)
        x32_2 = x32_2+self.module_list[93:95](x32_2)
        x32_2 = x32_2+self.module_list[96:98](x32_2)
        x32_2 = x32_2+self.module_list[99:101](x32_2)
        x32_2 = self.module_list[102](x32_2)
        x32 = torch.cat((x32_2,x32_1),1)
        x32 = self.module_list[104](x32)
        # ↑↑↑↑↑↑↑↑↑  backbone ↑↑↑↑↑↑↑↑↑↑↑

        x32 = self.module_list[105:108](x32)
        # spp
        x32 = torch.cat((x32,self.module_list[108](x32),self.module_list[110](x32),self.module_list[112](x32)),1)
        x32 = self.module_list[114](x32)

        _x32 = self.module_list[115:117](x32)
        x32 = self.module_list[117:119](_x32)
        x32 = torch.cat((self.module_list[120](x16),x32),1)

        _x32_1 = self.module_list[122:127](x32)
        x32 = self.module_list[127:129](_x32_1)
        x32 = torch.cat((self.module_list[130](x8),x32),1)

        _x32_2 = self.module_list[132:137](x32)

        x32 = self.module_list[137:139](_x32_2) # out

        x16 = torch.cat((self.module_list[141](_x32_2),_x32_1),1)
        _x16 = self.module_list[143:148](x16)
        x16 = self.module_list[148:150](_x16) # out

        x8 = torch.cat((self.module_list[152](_x16),_x32),1)
        x8 = self.module_list[154:161](x8)

        if self.transform:
            x8 = x8.permute(0, 2, 3, 1)  # (-1,7,7,30)
            bs, h, w, c = x8.shape
            x8 = x8.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

            x16 = x16.permute(0, 2, 3, 1)  # (-1,7,7,30)
            bs, h, w, c = x16.shape
            x16 = x16.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

            x32 = x32.permute(0, 2, 3, 1)  # (-1,7,7,30)
            bs, h, w, c = x32.shape
            x32 = x32.contiguous().view(bs, h, w, self.num_anchors, 5 + self.num_classes)

        return x32,x16,x8


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
    # model = Yolov3(num_classes=20,pretrained=True,pretrained_weights="./yolov3.pt")
    # model = Yolov3SPP(num_classes=20,pretrained=True,pretrained_weights="./yolov3-spp.pt")
    # # model = Yolov4(num_classes=20,pretrained=True,pretrained_weights="./yolov4.pt")
    # print(model)
    # pred = model(torch.rand([1,3,256,256]))
    # # print(pred.shape)

    path = "/media/wucong/225A6D42D4FA828F1/work/GitHub/toolkit/toolkit/papers/detection/YOLO/YOLOV3/yolov3_last/cfg/"
    path += "yolov5x.pt"
    model = Yolov5x(num_classes=20, pretrained=True, pretrained_weights=path)
    pred = model(torch.rand([1, 3, 256, 256]))