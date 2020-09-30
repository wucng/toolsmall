from torchvision.ops import misc as misc_nn_ops
from torchvision.models import resnet
from torch import nn
from collections import OrderedDict
import torchvision
import torch
from torch.nn import functional as F
import math
from torchvision.ops import roi_align,roi_pool

from .net import _initParmasV2

# --------------- FPN --------------------------
class FPNNet_BN(nn.Module):
    def __init__(self,in_channels_dict,out_channels=256):
        super().__init__()

        self.inner_blocks = nn.ModuleDict()
        self.layer_blocks = nn.ModuleDict()
        for name,in_channels in in_channels_dict.items():
            if in_channels == 0:
                continue
            inner_block_module = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                               nn.BatchNorm2d(out_channels),
                                               nn.ReLU(inplace=True))
            layer_block_module = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                               nn.BatchNorm2d(out_channels),
                                               nn.ReLU(inplace=True))
            self.inner_blocks[name] = inner_block_module
            self.layer_blocks[name] = layer_block_module

        # initialize parameters now to avoid modifying the initialization of top_blocks
        # _initParmas(self, self.modules())
        _initParmasV2(self, self.modules())
        # _initParmas2(self,self.named_parameters())


    def forward(self,features):
        # outNmae="" # "p"
        outs={}
        last_inner = None
        name = ""
        for i,name in enumerate(sorted(features)[::-1]): # 至上而下  [C5,C4,C3,C2]
            inner_lateral = self.inner_blocks[name](features[name])
            feat_shape = inner_lateral.shape[-2:]
            if last_inner is None:
                last_inner = inner_lateral
            else:
                inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
                last_inner = inner_lateral + inner_top_down
            outs[name] = self.layer_blocks[name](last_inner)

        # 最后一个做pool 只用于提取框
        # outs["pool"]=F.max_pool2d(outs[name], 1, 2, 0) # "pool"

        return outs

class FPNNet_BNT(nn.Module): # 至上而下 # [P7,P6,P5,...P2]
    def __init__(self, in_channels_dict, out_channels=256,mode="concat"):
        super().__init__()
        assert mode in ["concat","add"]
        self.mode = mode
        self.net = nn.ModuleDict()
        self.reduce = nn.ModuleDict()

        keys = sorted(list(in_channels_dict.keys()))[::-1] # [P7,P6,P5,...P2]
        for i, key in enumerate(keys):
            in_channels = in_channels_dict[key]
            if i == 0:
                self.reduce[key] = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                self.net[key] = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(out_channels, out_channels, 3, 2, 1, 1),  # 2,2,0,0
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
                if self.mode == "concat":
                    self.reduce[key] = nn.Sequential(
                        nn.Conv2d(in_channels + out_channels, out_channels, 3, 1, 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                else:
                    self.reduce[key] = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )

        _initParmasV2(self, self.modules())

    def forward(self, features):
        outs = {}
        keys = sorted(list(features.keys()))[::-1]  # [P7,P6,P5,...P2]
        for i, name in enumerate(keys):
            if i == 0:
                outs[name] = self.reduce[name](features[name])
            else:
                if self.mode == "concat":
                    x = torch.cat((features[name], self.net[name](outs[keys[i - 1]])),1)
                    x = self.reduce[name](x)
                else:
                    x = self.reduce[name](features[name]) + self.net[name](outs[keys[i - 1]])
                outs[name] = x

        return outs

class PANet_DTU(nn.Module): # down to up  至下而上 [p2,p3,...p7]
    def __init__(self, in_channels_dict, out_channels=256,mode="concat"):
        super().__init__()
        assert mode in ["concat","add"]
        self.mode = mode
        self.net = nn.ModuleDict()
        self.reduce = nn.ModuleDict()

        keys = sorted(list(in_channels_dict.keys())) # [p2,p3,...p7]
        for i, key in enumerate(keys):
            in_channels = in_channels_dict[key]
            if i == 0:
                self.reduce[key] = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                self.net[key] = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 2, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
                if self.mode == "concat":
                    self.reduce[key] = nn.Sequential(
                        nn.Conv2d(in_channels + out_channels, out_channels, 3, 1, 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                else:
                    self.reduce[key] = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )

        _initParmasV2(self, self.modules())

    def forward(self, features):
        outs = {}
        keys = sorted(list(features.keys()))  # [p2,p3,...p7]
        for i, name in enumerate(keys):
            if i == 0:
                outs[name] = self.reduce[name](features[name])
            else:
                if self.mode == "concat":
                    x = torch.cat((features[name], self.net[name](outs[keys[i - 1]])),1)
                    x = self.reduce[name](x)
                else:
                    x = self.reduce[name](features[name]) + self.net[name](outs[keys[i - 1]])
                outs[name] = x

        return outs

# PANet = FPN + PANet_DTU