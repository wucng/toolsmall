from torchvision.ops import misc as misc_nn_ops
from torchvision.models import resnet
from torch import nn
from collections import OrderedDict
import torchvision
import torch
from torch.nn import functional as F
import math
from torchvision.ops import roi_align,roi_pool

# import torch
# from torch import nn
# from torch.nn import functional as F
from torchvision.models.vgg import load_state_dict_from_url,model_urls

try:
    from .net import _initParmasV2,_initialize_weights
except:
    from net import _initParmasV2,_initialize_weights

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


import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.vgg import load_state_dict_from_url,model_urls

class Vgg16BN(nn.Module):
    def __init__(self,in_channels=3,return_indices=False,init_weights=True):
        super().__init__()
        self.return_indices = return_indices
        self.inplanes = 64

        layer1 = nn.Sequential(
            nn.Conv2d(in_channels,self.inplanes,3,padding=1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes, self.inplanes, 3, padding=1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, return_indices=return_indices)
        )

        layer2 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes*2, 3, padding=1),
            nn.BatchNorm2d(self.inplanes*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes*2, self.inplanes*2, 3, padding=1),
            nn.BatchNorm2d(self.inplanes*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, return_indices=return_indices)
        )

        layer3 = nn.Sequential(
            nn.Conv2d(self.inplanes*2, self.inplanes * 4, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes * 4, self.inplanes * 4, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes * 4, self.inplanes * 4, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, return_indices=return_indices)
        )

        layer4 = nn.Sequential(
            nn.Conv2d(self.inplanes * 4, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, return_indices=return_indices)
        )

        layer5 = nn.Sequential(
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, return_indices=return_indices)
        )

        # self.features.add_module("",layer1)
        self.features = nn.Sequential(*[*layer1,*layer2,*layer3,*layer4,*layer5])

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # def forward(self,x):
    #     out = {}
    #     indices = {}
    #     if self.return_indices:
    #         x1,indices1 = self.features[0:7](x)
    #         x2,indices2 = self.features[7:14](x1)
    #         x3,indices3 = self.features[14:24](x2)
    #         x4,indices4 = self.features[24:34](x3)
    #         x5,indices5 = self.features[34:](x4)
    #         out["res1"] = x1
    #         out["res2"] = x2
    #         out["res3"] = x3
    #         out["res4"] = x4
    #         out["res5"] = x5
    #
    #         indices["res1"] = indices1
    #         indices["res2"] = indices2
    #         indices["res3"] = indices3
    #         indices["res4"] = indices4
    #         indices["res5"] = indices5
    #     else:
    #         x1 = self.features[0:7](x)
    #         x2 = self.features[7:14](x1)
    #         x3 = self.features[14:24](x2)
    #         x4 = self.features[24:34](x3)
    #         x5 = self.features[34:](x4)
    #         out["res1"] = x1
    #         out["res2"] = x2
    #         out["res3"] = x3
    #         out["res4"] = x4
    #         out["res5"] = x5
    #
    #     return out,indices


def vgg16BN(pretrained=False,**kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = Vgg16BN(**kwargs)
    if pretrained:
        # 加载预训练的模型
        state_dict = load_state_dict_from_url(model_urls["vgg16_bn"], progress=True)
        model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()})
    return model


class Backbone_vgg16bn(nn.Module):
    def __init__(self,model_name="vgg16_bn",pretrained=False,freeze_at=["res1","res2","res3","res4","res5"],return_indices=False):
        super().__init__()
        self.return_indices = return_indices

        _model = vgg16BN(pretrained, in_channels=3, return_indices=return_indices, init_weights=True)
        self.inplanes = _model.inplanes
        self.out_channels = self.inplanes * 8

        self.backbone = nn.ModuleDict(OrderedDict([  # nn.Sequential
            ("res1", _model.features[0:7]),
            ("res2", _model.features[7:14]),
            ("res3", _model.features[14:24]),
            ("res4", _model.features[24:34]),
            ("res5", _model.features[34:]),
        ]))

        # 参数冻结
        for name in freeze_at:
            for parameter in self.backbone[name].parameters():
                parameter.requires_grad_(False)

        # 统计所有可更新梯度的变量
        print("只有以下变量做梯度更新:")
        for name, parameter in self.backbone.named_parameters():
            if parameter.requires_grad:
                print("name:", name)

    def forward(self,x):
        out = {}
        indices = {}
        if self.return_indices:
            x1, indices1 = self.backbone.res1(x)
            x2, indices2 = self.backbone.res2(x1)
            x3, indices3 = self.backbone.res3(x2)
            x4, indices4 = self.backbone.res4(x3)
            x5, indices5 = self.backbone.res5(x4)
            out["res1"] = x1
            out["res2"] = x2
            out["res3"] = x3
            out["res4"] = x4
            out["res5"] = x5

            indices["res1"] = indices1
            indices["res2"] = indices2
            indices["res3"] = indices3
            indices["res4"] = indices4
            indices["res5"] = indices5

            return out,indices
        else:
            x1 = self.backbone.res1(x)
            x2 = self.backbone.res2(x1)
            x3 = self.backbone.res3(x2)
            x4 = self.backbone.res4(x3)
            x5 = self.backbone.res5(x4)
            out["res1"] = x1
            out["res2"] = x2
            out["res3"] = x3
            out["res4"] = x4
            out["res5"] = x5

            return out


class Backbone_vgg(nn.Module):
    def __init__(self,model_name="vgg16_bn",pretrained=False,freeze_at=["res1","res2","res3","res4","res5"]):
        super().__init__()
        _model = torchvision.models.vgg.__dict__[model_name](pretrained=pretrained)

        self.inplanes = 64
        self.out_channels = 8*self.inplanes

        self.backbone = nn.ModuleDict(OrderedDict([  # nn.Sequential
            ("res1", _model.features[0:7]),
            ("res2", _model.features[7:14]),
            ("res3", _model.features[14:24]),
            ("res4", _model.features[24:34]),
            ("res5", _model.features[34:44]),
        ]))

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

        x1 = self.backbone.res1(x)
        x2 = self.backbone.res2(x1)
        x3 = self.backbone.res3(x2)
        x4 = self.backbone.res4(x3)
        x5 = self.backbone.res5(x4)
        out["res1"] = x1
        out["res2"] = x2
        out["res3"] = x3
        out["res4"] = x4
        out["res5"] = x5

        return out

"""
参考 segnet:https://arxiv.org/pdf/1511.00561v3.pdf
"""
class Vgg16BNDecode(nn.Module):
    def __init__(self,num_classes=21,freeze_at=["res2","res3","res4","res5"],use_shortcut=False,pretrained=False,return_indices=True):
        super().__init__()

        self.num_classes = num_classes
        self.use_shortcut = use_shortcut

        self.backbone = Backbone_vgg16bn(pretrained=pretrained,freeze_at=freeze_at,return_indices=return_indices)

        self.inplanes = self.backbone.inplanes

        self.decode = nn.ModuleDict()
        self.decode["res5"]=nn.Sequential(
            # nn.MaxUnpool2d(2,2),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=False)
        )
        self.decode["res4"] = nn.Sequential(
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 4, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 4),
            nn.ReLU(inplace=False)
        )

        self.decode["res3"] = nn.Sequential(
            nn.Conv2d(self.inplanes * 4, self.inplanes * 4, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes * 4, self.inplanes * 4, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes * 4, self.inplanes * 2, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 2),
            nn.ReLU(inplace=False)
        )

        self.decode["res2"] = nn.Sequential(
            nn.Conv2d(self.inplanes*2, self.inplanes*2, 3, padding=1),
            nn.BatchNorm2d(self.inplanes*2),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes*2, self.inplanes, 3, padding=1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=False)
        )

        self.decode["res1"] = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes, 3, padding=1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes, num_classes, 3, padding=1),
            # nn.BatchNorm2d(num_classes),
            # nn.ReLU(inplace=False)
            # nn.Softmax(1)
        )

        _initParmasV2(self,self.decode.modules())

    def forward(self,x):
        out,indices = self.backbone(x)

        x5 = self.decode["res5"](F.max_unpool2d(out["res5"], indices["res5"], 2, 2))
        if self.use_shortcut:
            x5 += out["res4"]

        x4 = self.decode["res4"](F.max_unpool2d(x5, indices["res4"], 2, 2))
        if self.use_shortcut:
            x4 += out["res3"]

        x3 = self.decode["res3"](F.max_unpool2d(x4, indices["res3"], 2, 2))
        if self.use_shortcut:
            x3 += out["res2"]

        x2 = self.decode["res2"](F.max_unpool2d(x3, indices["res2"], 2, 2))
        if self.use_shortcut:
            x2 += out["res1"]

        x1 = self.decode["res1"](F.max_unpool2d(x2, indices["res1"], 2, 2))

        x1 = x1.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
        return x1


class Vgg16BNDecodeV2(nn.Module):
    def __init__(self,model_name="vgg16_bn",num_classes=21,freeze_at=["res2","res3","res4","res5"],use_shortcut=False,pretrained=False,return_indices=False):
        super().__init__()

        self.num_classes = num_classes
        self.use_shortcut = use_shortcut

        # self.backbone = Backbone_vgg16bn(pretrained=pretrained,freeze_at=freeze_at,return_indices=return_indices)
        self.backbone = Backbone_vgg(model_name,pretrained=pretrained,freeze_at=freeze_at)

        self.inplanes = self.backbone.inplanes

        self.decode = nn.ModuleDict()
        self.decode["res5"]=nn.Sequential(
            # nn.MaxUnpool2d(2,2),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=False)
        )
        self.decode["res4"] = nn.Sequential(
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 4, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 4),
            nn.ReLU(inplace=False)
        )

        self.decode["res3"] = nn.Sequential(
            nn.Conv2d(self.inplanes * 4, self.inplanes * 4, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes * 4, self.inplanes * 4, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes * 4, self.inplanes * 2, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 2),
            nn.ReLU(inplace=False)
        )

        self.decode["res2"] = nn.Sequential(
            nn.Conv2d(self.inplanes*2, self.inplanes*2, 3, padding=1),
            nn.BatchNorm2d(self.inplanes*2),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes*2, self.inplanes, 3, padding=1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=False)
        )

        self.decode["res1"] = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes, 3, padding=1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes, num_classes, 3, padding=1),
            # nn.BatchNorm2d(num_classes),
            # nn.ReLU(inplace=False)
            # nn.Softmax(1)
        )

        _initParmasV2(self,self.decode.modules())

    def forward(self,x):
        out = self.backbone(x)

        x5 = self.decode["res5"](F.interpolate(out["res5"],scale_factor=2))
        if self.use_shortcut:
            x5 += out["res4"]

        x4 = self.decode["res4"](F.interpolate(x5, scale_factor=2))
        if self.use_shortcut:
            x4 += out["res3"]

        x3 = self.decode["res3"](F.interpolate(x4, scale_factor=2))
        if self.use_shortcut:
            x3 += out["res2"]

        x2 = self.decode["res2"](F.interpolate(x3, scale_factor=2))
        if self.use_shortcut:
            x2 += out["res1"]

        x1 = self.decode["res1"](F.interpolate(x2, scale_factor=2))

        x1 = x1.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
        return x1


# 从头实现（return_indices=True 就是 Vgg16BNDecode ；return_indices=False 就是 Vgg16BNDecodeV2）
class Vgg16BNDecodeV3(nn.Module): # use_shortcut=True 会报错 ？
    def __init__(self,model_name="vgg16_bn",num_classes=21,freeze_at=["res2","res3","res4","res5"],use_shortcut=False,pretrained=False,return_indices=False):
        super().__init__()
        self.return_indices = return_indices
        self.num_classes = num_classes
        self.use_shortcut = use_shortcut
        self.inplanes = 64
        features = nn.Sequential(
            nn.Conv2d(3, self.inplanes, 3, padding=1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes, self.inplanes, 3, padding=1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, return_indices=return_indices),

            nn.Conv2d(self.inplanes, self.inplanes * 2, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes * 2, self.inplanes * 2, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, return_indices=return_indices),

            nn.Conv2d(self.inplanes * 2, self.inplanes * 4, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes * 4, self.inplanes * 4, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes * 4, self.inplanes * 4, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, return_indices=return_indices),

            nn.Conv2d(self.inplanes * 4, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, return_indices=return_indices),

            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, return_indices=return_indices)
        )

        # self._initialize_weights()
        if pretrained:
            # 加载预训练的模型
            state_dict = load_state_dict_from_url(model_urls["vgg16_bn"], progress=True)
            features.load_state_dict({k[9:]: v for k, v in state_dict.items() if k[9:] in features.state_dict()})
        else:
            _initialize_weights(self,features.modules())

        self.backbone = nn.ModuleDict(OrderedDict([  # nn.Sequential
            ("res1", features[0:7]),
            ("res2", features[7:14]),
            ("res3", features[14:24]),
            ("res4", features[24:34]),
            ("res5", features[34:]),
        ]))

        # 参数冻结
        for name in freeze_at:
            for parameter in self.backbone[name].parameters():
                parameter.requires_grad_(False)

        # 统计所有可更新梯度的变量
        print("只有以下变量做梯度更新:")
        for name, parameter in self.backbone.named_parameters():
            if parameter.requires_grad:
                print("name:", name)

        self.decode = nn.ModuleDict()
        self.decode["res5"] = nn.Sequential(
            # nn.MaxUnpool2d(2,2),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=False)
        )
        self.decode["res4"] = nn.Sequential(
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 8, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes * 8, self.inplanes * 4, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 4),
            nn.ReLU(inplace=False)
        )

        self.decode["res3"] = nn.Sequential(
            nn.Conv2d(self.inplanes * 4, self.inplanes * 4, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes * 4, self.inplanes * 4, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes * 4, self.inplanes * 2, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 2),
            nn.ReLU(inplace=False)
        )

        self.decode["res2"] = nn.Sequential(
            nn.Conv2d(self.inplanes * 2, self.inplanes * 2, 3, padding=1),
            nn.BatchNorm2d(self.inplanes * 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes * 2, self.inplanes, 3, padding=1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=False)
        )

        self.decode["res1"] = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes, 3, padding=1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.inplanes, num_classes, 3, padding=1),
            # nn.BatchNorm2d(num_classes),
            # nn.ReLU(inplace=False)
            # nn.Softmax(1)
        )

        _initParmasV2(self, self.decode.modules())

    def forward(self,x):
        if self.return_indices:
            # encode
            x1, indices1 = self.backbone.res1(x)
            x2, indices2 = self.backbone.res2(x1)
            x3, indices3 = self.backbone.res3(x2)
            x4, indices4 = self.backbone.res4(x3)
            x5, indices5 = self.backbone.res5(x4)

            # decode
            dx5 = self.decode.res5(F.max_unpool2d(x5, indices5, 2, 2))
            if self.use_shortcut:
                dx5 += x4

            dx4 = self.decode.res4(F.max_unpool2d(dx5, indices4, 2, 2))
            if self.use_shortcut:
                dx4 += x3

            dx3 = self.decode.res3(F.max_unpool2d(dx4, indices3, 2, 2))
            if self.use_shortcut:
                dx3 += x2

            dx2 = self.decode.res2(F.max_unpool2d(dx3, indices2, 2, 2))
            if self.use_shortcut:
                dx2 += x1

            dx1 = self.decode.res1(F.max_unpool2d(dx2, indices1, 2, 2))

        else:
            x1 = self.backbone.res1(x)
            x2 = self.backbone.res2(x1)
            x3 = self.backbone.res3(x2)
            x4 = self.backbone.res4(x3)
            x5 = self.backbone.res5(x4)

            # decode
            dx5 = self.decode.res5(F.interpolate(x5, scale_factor=2))
            if self.use_shortcut:
                dx5 += x4

            dx4 = self.decode.res4(F.interpolate(dx5, scale_factor=2))
            if self.use_shortcut:
                dx4 += x3

            dx3 = self.decode.res3(F.interpolate(dx4, scale_factor=2))
            if self.use_shortcut:
                dx3 += x2

            dx2 = self.decode.res2(F.interpolate(dx3, scale_factor=2))
            if self.use_shortcut:
                dx2 += x1

            dx1 = self.decode.res1(F.interpolate(dx2, scale_factor=2))

        dx1 = dx1.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)

        return dx1


if __name__=="__main__":
    x = torch.rand([1,3,224,224])
    # model = vgg16BN(True,return_indices=True)
    # pred = model(x)
    # print(pred.shape)

    model = Vgg16BNDecodeV3(num_classes=21,pretrained=True,use_shortcut=True,return_indices=True)
    pred = model(x)
    print()

    # x = torch.rand([1, 1, 4, 4]).clone()
    # print(x.squeeze())
    # y, indices = nn.MaxPool2d(2, 2, 0, return_indices=True)(x)
    # print(y)
    #
    # y = torch.rand([1, 1, 2, 2])
    # x1 = nn.MaxUnpool2d(2, 2, 0)(y, indices)
    # print(x1)