from torchvision.ops import misc as misc_nn_ops
from torchvision.models import resnet
from torch import nn
from collections import OrderedDict
import torchvision
import torch
from torch.nn import functional as F
import math
from torchvision.ops import roi_align,roi_pool

try:
    from .backbone import SeBlock,CBAM,resnetv2,resnetv3,resnetv4,dlav0,dlav1
except:
    from backbone import SeBlock, CBAM, resnetv2, resnetv3, resnetv4,dlav0,dlav1

__all__=["Backbone","ResnetFpn","RPNHead","TwoMLPHead","FastRCNNPredictor"]

def _initParmas(self,modules):
    # for m in self.children():
    #     if isinstance(m, nn.Conv2d):
    #         # nn.init.kaiming_uniform_(m.weight, a=1)
    #         nn.init.normal_(m.weight, 0.01)
    #         nn.init.constant_(m.bias, 0)

    for m in modules:
        if isinstance(m, (nn.Conv2d,nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, std=0.01)
            # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            # if m.bias is not None:
            #     nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            # if m.bias is not None:
                # nn.init.zeros_(m.bias)
                # nn.init.constant_(m.bias, 0)

def _initParmasV2(self,modules):

    for m in modules:
        if isinstance(m, (nn.Conv2d,nn.ConvTranspose2d)):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            # if m.bias is not None:
            #     nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            # if m.bias is not None:
                # nn.init.zeros_(m.bias)
                # nn.init.constant_(m.bias, 0)

def _initParmas2(self,named_parameters):
    for name, param in named_parameters:
        if "weight" in name:
            try:  # Conv2d
                # nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
                nn.init.normal_(param, 0.01)
            except:  # BN
                nn.init.constant_(param, 1)
        elif "bias" in name:
            nn.init.constant_(param, 0)



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

class Backbone(nn.Module):
    def __init__(self,model_name="resnet18",pretrained=False,nofreeze_at=["res2","res3","res4","res5"]):# ["res1","res2","res3","res4","res5"]
        super(Backbone, self).__init__()
        model_dict = {'resnet18': 512,
                      'resnet34': 512,
                      'resnet50': 2048,
                      'resnet101': 2048,
                      'resnet152': 2048,
                      'resnext50_32x4d': 2048,
                      'resnext101_32x8d': 2048,
                      'wide_resnet50_2': 2048,
                      'wide_resnet101_2': 2048}

        assert model_name in model_dict, "%s must be in %s" % (model_name, model_dict.keys())

        self.out_channels = model_dict[model_name]
        _model = torchvision.models.resnet.__dict__[model_name](pretrained=pretrained)
        # backbone_size = _model.inplanes

        layer0 = nn.Sequential(
            _model.conv1,
            _model.bn1,
            _model.relu,
            _model.maxpool
        )
        self.backbone = nn.Sequential(OrderedDict([
            ("res1",layer0),
            ("res2", _model.layer1),
            ("res3", _model.layer2),
            ("res4", _model.layer3),
            ("res5", _model.layer4),
        ]))

        # freeze layers (layer1)
        for name, parameter in self.backbone.named_parameters():
            # if 'res3' not in name and 'res4' not in name and 'res5' not in name:
            flag = True
            for nofreezename in nofreeze_at:
                if nofreezename in name:
                    flag = False
                    break
            if flag:
                parameter.requires_grad_(False)

        # 默认冻结 BN中的参数 不更新
        # for m in self.backbone.modules():
        #     if isinstance(m,(nn.BatchNorm2d, nn.GroupNorm)):
        #        for parameter in m.parameters():
        #            parameter.requires_grad_(False)


    def forward(self,x):
        # x = self.backbone(x)
        out={}
        x = self.backbone.res1(x)
        x = self.backbone.res2(x)
        out["res2"]=x
        x = self.backbone.res3(x)
        out["res3"] = x
        x = self.backbone.res4(x)
        out["res4"] = x
        x = self.backbone.res5(x)
        out["res5"] = x

        return out

class BackboneV2(nn.Module):
    def __init__(self,model_name="resnet18",pretrained=False,freeze_at=["res1","res2","res3","res4","res5"]):# ["res1","res2","res3","res4","res5"]
        super().__init__()
        model_dict = {'resnet18': 512,
                      'resnet34': 512,
                      'resnet50': 2048,
                      'resnet101': 2048,
                      'resnet152': 2048,
                      'resnext50_32x4d': 2048,
                      'resnext101_32x8d': 2048,
                      'wide_resnet50_2': 2048,
                      'wide_resnet101_2': 2048}

        assert model_name in model_dict, "%s must be in %s" % (model_name, model_dict.keys())

        self.out_channels = model_dict[model_name]
        _model = torchvision.models.resnet.__dict__[model_name](pretrained=pretrained)
        # backbone_size = _model.inplanes

        layer0 = nn.Sequential(
            _model.conv1,
            _model.bn1,
            _model.relu,
            _model.maxpool
        )
        self.backbone = nn.ModuleDict(OrderedDict([ # nn.Sequential
            ("res1",layer0),
            ("res2", _model.layer1),
            ("res3", _model.layer2),
            ("res4", _model.layer3),
            ("res5", _model.layer4),
        ]))


        # 参数冻结
        for name in freeze_at:
            for parameter in self.backbone[name].parameters():
                parameter.requires_grad_(False)

        # 统计所有可更新梯度的变量
        print("只有以下变量做梯度更新:")
        for name,parameter in self.backbone.named_parameters():
            if parameter.requires_grad:
                print("name:",name)

        # 默认冻结 BN中的参数 不更新
        # for m in self.backbone.modules():
        #     if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         for parameter in m.parameters():
        #             parameter.requires_grad_(False)

    def forward(self,x):
        out={}
        x = self.backbone.res1(x)
        x = self.backbone.res2(x)
        out["res2"]=x
        x = self.backbone.res3(x)
        out["res3"] = x
        x = self.backbone.res4(x)
        out["res4"] = x
        x = self.backbone.res5(x)
        out["res5"] = x

        return out

# 增加贯通层 26 × 26 × 512-->13 × 13 × 2048
class BackboneV2_yolov2(nn.Module):
    def __init__(self,model_name="resnet18",pretrained=False,freeze_at=["res1","res2","res3","res4","res5"]):# ["res1","res2","res3","res4","res5"]
        super().__init__()
        model_dict = {'resnet18': 512,
                      'resnet34': 512,
                      'resnet50': 2048,
                      'resnet101': 2048,
                      'resnet152': 2048,
                      'resnext50_32x4d': 2048,
                      'resnext101_32x8d': 2048,
                      'wide_resnet50_2': 2048,
                      'wide_resnet101_2': 2048}

        assert model_name in model_dict, "%s must be in %s" % (model_name, model_dict.keys())

        self.out_channels = model_dict[model_name]
        _model = torchvision.models.resnet.__dict__[model_name](pretrained=pretrained)
        # backbone_size = _model.inplanes

        layer0 = nn.Sequential(
            _model.conv1,
            _model.bn1,
            _model.relu,
            _model.maxpool
        )
        self.backbone = nn.ModuleDict(OrderedDict([ # nn.Sequential
            ("res1",layer0),
            ("res2", _model.layer1),
            ("res3", _model.layer2),
            ("res4", _model.layer3),
            ("res5", _model.layer4),
        ]))

        # 参数冻结
        for name in freeze_at:
            for parameter in self.backbone[name].parameters():
                parameter.requires_grad_(False)

        # 统计所有可更新梯度的变量
        print("只有以下变量做梯度更新:")
        for name,parameter in self.backbone.named_parameters():
            if parameter.requires_grad:
                print("name:",name)

        # 默认冻结 BN中的参数 不更新
        # for m in self.backbone.modules():
        #     if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         for parameter in m.parameters():
        #             parameter.requires_grad_(False)

    def forward(self,x):
        out={}
        x = self.backbone.res1(x)
        x = self.backbone.res2(x)
        out["res2"]=x
        x = self.backbone.res3(x)
        out["res3"] = x
        x = self.backbone.res4(x)
        out["res4"] = x
        x = self.backbone.res5(x)
        # out["res5"] = x

        # passthrough layer
        # 增加贯通层 26 × 26 × 512-->13 × 13 × 2048
        bs,c,h,w = out["res4"].shape
        x1 = out["res4"].contiguous().view(bs,-1,h//2,w//2)
        x = torch.cat((x1,x),1)
        out["res5"] = x

        return out


class BackboneV2_yolov3(nn.Module):
    def __init__(self,model_name="resnet18",pretrained=False,freeze_at=["res1","res2","res3","res4","res5"]):# ["res1","res2","res3","res4","res5"]
        super().__init__()
        model_dict = {'resnet18': 512,
                      'resnet34': 512,
                      'resnet50': 2048,
                      'resnet101': 2048,
                      'resnet152': 2048,
                      'resnext50_32x4d': 2048,
                      'resnext101_32x8d': 2048,
                      'wide_resnet50_2': 2048,
                      'wide_resnet101_2': 2048}

        assert model_name in model_dict, "%s must be in %s" % (model_name, model_dict.keys())

        self.out_channels = model_dict[model_name]
        _model = torchvision.models.resnet.__dict__[model_name](pretrained=pretrained)
        # backbone_size = _model.inplanes

        layer0 = nn.Sequential(
            _model.conv1,
            _model.bn1,
            _model.relu,
            _model.maxpool
        )
        self.backbone = nn.ModuleDict(OrderedDict([ # nn.Sequential
            ("res1",layer0),
            ("res2", _model.layer1),
            ("res3", _model.layer2),
            ("res4", _model.layer3),
            ("res5", _model.layer4),
        ]))


        # 参数冻结
        for name in freeze_at:
            for parameter in self.backbone[name].parameters():
                parameter.requires_grad_(False)

        # 统计所有可更新梯度的变量
        print("只有以下变量做梯度更新:")
        for name,parameter in self.backbone.named_parameters():
            if parameter.requires_grad:
                print("name:",name)

        # 默认冻结 BN中的参数 不更新
        # for m in self.backbone.modules():
        #     if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         for parameter in m.parameters():
        #             parameter.requires_grad_(False)

    def forward(self,x):
        out={}
        x = self.backbone.res1(x)
        x = self.backbone.res2(x)
        out["res2"]=x
        x = self.backbone.res3(x)
        out["res3"] = x
        x = self.backbone.res4(x)
        out["res4"] = x
        x = self.backbone.res5(x)
        out["res5"] = x

        out["res3"] = torch.cat((out["res3"], F.interpolate(out["res4"], scale_factor=2.0, mode="nearest")), 1)
        out["res4"] = torch.cat((out["res4"],F.interpolate(out["res5"], scale_factor=2.0, mode="nearest")),1)

        return out

# 整个backbone网络 stride=16
class Backbone_s16(nn.Module):
    def __init__(self,model_name="resnet18",pretrained=False,freeze_at=["res1","res2","res3","res4","res5"]):# ["res1","res2","res3","res4","res5"]
        super().__init__()
        model_dict = {'resnet18': 512,
                      'resnet34': 512,
                      'resnet50': 2048,
                      'resnet101': 2048,
                      'resnet152': 2048,
                      'resnext50_32x4d': 2048,
                      'resnext101_32x8d': 2048,
                      'wide_resnet50_2': 2048,
                      'wide_resnet101_2': 2048}

        assert model_name in model_dict, "%s must be in %s" % (model_name, model_dict.keys())

        self.out_channels = model_dict[model_name]
        # _model = torchvision.models.resnet.__dict__[model_name](pretrained=pretrained)
        _model = resnetv2.__dict__[model_name](pretrained=pretrained)
        # backbone_size = _model.inplanes

        layer0 = nn.Sequential(
            _model.conv1,
            _model.bn1,
            _model.relu,
            _model.maxpool
        )
        self.backbone = nn.ModuleDict(OrderedDict([ # nn.Sequential
            ("res1",layer0),
            ("res2", _model.layer1),
            ("res3", _model.layer2),
            ("res4", _model.layer3),
            ("res5", _model.layer4),
        ]))


        # 参数冻结
        for name in freeze_at:
            for parameter in self.backbone[name].parameters():
                parameter.requires_grad_(False)

        # 统计所有可更新梯度的变量
        print("只有以下变量做梯度更新:")
        for name,parameter in self.backbone.named_parameters():
            if parameter.requires_grad:
                print("name:",name)

        # 默认冻结 BN中的参数 不更新
        # for m in self.backbone.modules():
        #     if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         for parameter in m.parameters():
        #             parameter.requires_grad_(False)

    def forward(self,x):
        out={}
        x = self.backbone.res1(x)
        x = self.backbone.res2(x)
        out["res2"]=x
        x = self.backbone.res3(x)
        out["res3"] = x
        x = self.backbone.res4(x)
        out["res4"] = x
        x = self.backbone.res5(x)
        out["res5"] = x

        return out

class Backbone_s8(nn.Module):
    def __init__(self,model_name="resnet18",pretrained=False,freeze_at=["res1","res2","res3","res4","res5"]):# ["res1","res2","res3","res4","res5"]
        super().__init__()
        model_dict = {'resnet18': 512,
                      'resnet34': 512,
                      'resnet50': 2048,
                      'resnet101': 2048,
                      'resnet152': 2048,
                      'resnext50_32x4d': 2048,
                      'resnext101_32x8d': 2048,
                      'wide_resnet50_2': 2048,
                      'wide_resnet101_2': 2048}

        assert model_name in model_dict, "%s must be in %s" % (model_name, model_dict.keys())

        self.out_channels = model_dict[model_name]
        # _model = torchvision.models.resnet.__dict__[model_name](pretrained=pretrained)
        _model = resnetv3.__dict__[model_name](pretrained=pretrained)
        # backbone_size = _model.inplanes

        layer0 = nn.Sequential(
            _model.conv1,
            _model.bn1,
            _model.relu,
            _model.maxpool
        )
        self.backbone = nn.ModuleDict(OrderedDict([ # nn.Sequential
            ("res1",layer0),
            ("res2", _model.layer1),
            ("res3", _model.layer2),
            ("res4", _model.layer3),
            ("res5", _model.layer4),
        ]))


        # 参数冻结
        for name in freeze_at:
            for parameter in self.backbone[name].parameters():
                parameter.requires_grad_(False)

        # 统计所有可更新梯度的变量
        print("只有以下变量做梯度更新:")
        for name,parameter in self.backbone.named_parameters():
            if parameter.requires_grad:
                print("name:",name)

        # 默认冻结 BN中的参数 不更新
        # for m in self.backbone.modules():
        #     if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         for parameter in m.parameters():
        #             parameter.requires_grad_(False)

    def forward(self,x):
        out={}
        x = self.backbone.res1(x)
        x = self.backbone.res2(x)
        out["res2"]=x
        x = self.backbone.res3(x)
        out["res3"] = x
        x = self.backbone.res4(x)
        out["res4"] = x
        x = self.backbone.res5(x)
        out["res5"] = x

        return out

class Backbone_s4(nn.Module):
    def __init__(self,model_name="resnet18",pretrained=False,freeze_at=["res1","res2","res3","res4","res5"]):# ["res1","res2","res3","res4","res5"]
        super().__init__()
        model_dict = {'resnet18': 512,
                      'resnet34': 512,
                      'resnet50': 2048,
                      'resnet101': 2048,
                      'resnet152': 2048,
                      'resnext50_32x4d': 2048,
                      'resnext101_32x8d': 2048,
                      'wide_resnet50_2': 2048,
                      'wide_resnet101_2': 2048}

        assert model_name in model_dict, "%s must be in %s" % (model_name, model_dict.keys())

        self.out_channels = model_dict[model_name]
        # _model = torchvision.models.resnet.__dict__[model_name](pretrained=pretrained)
        _model = resnetv4.__dict__[model_name](pretrained=pretrained)
        # backbone_size = _model.inplanes

        layer0 = nn.Sequential(
            _model.conv1,
            _model.bn1,
            _model.relu,
            _model.maxpool
        )
        self.backbone = nn.ModuleDict(OrderedDict([ # nn.Sequential
            ("res1",layer0),
            ("res2", _model.layer1),
            ("res3", _model.layer2),
            ("res4", _model.layer3),
            ("res5", _model.layer4),
        ]))


        # 参数冻结
        for name in freeze_at:
            for parameter in self.backbone[name].parameters():
                parameter.requires_grad_(False)

        # 统计所有可更新梯度的变量
        print("只有以下变量做梯度更新:")
        for name,parameter in self.backbone.named_parameters():
            if parameter.requires_grad:
                print("name:",name)

        # 默认冻结 BN中的参数 不更新
        # for m in self.backbone.modules():
        #     if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         for parameter in m.parameters():
        #             parameter.requires_grad_(False)

    def forward(self,x):
        out={}
        x = self.backbone.res1(x)
        x = self.backbone.res2(x)
        out["res2"]=x
        x = self.backbone.res3(x)
        out["res3"] = x
        x = self.backbone.res4(x)
        out["res4"] = x
        x = self.backbone.res5(x)
        out["res5"] = x

        return out

"""不推荐"""
class FPNNet(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPNNet, self).__init__()

        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        _initParmasV2(self,self.modules())
        # _initParmas2(self,self.named_parameters())

    def forward(self,features):
        # outNmae="" # "p"
        outs={}
        last_inner = None
        i = 0
        for i,name in enumerate(sorted(features)[::-1]): # [C5,C4,C3,C2]
            inner_lateral = self.inner_blocks[i](features[name])
            feat_shape = inner_lateral.shape[-2:]
            if last_inner is None:
                last_inner = inner_lateral
            else:
                inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
                last_inner = inner_lateral + inner_top_down
            outs[i] = self.layer_blocks[i](last_inner)

        # 最后一个做pool 只用于提取框
        outs[4]=F.max_pool2d(outs[i], 1, 2, 0) # "pool"

        return outs

class FPNNetV2(nn.Module):
    def __init__(self,in_channels_dict,out_channels):
        super(FPNNetV2, self).__init__()

        self.inner_blocks = nn.ModuleDict()
        self.layer_blocks = nn.ModuleDict()
        for name,in_channels in in_channels_dict.items():
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks[name] = inner_block_module
            self.layer_blocks[name] = (layer_block_module)

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

class ResnetFpnV2(nn.Module):
    def __init__(self,model_name,pretrained=False,freeze_at=["res1","res2","res3","res4","res5"],
                 out_channels=256,useFPN=False):
        super(ResnetFpnV2,self).__init__()
        self.backbone = BackboneV2(model_name,pretrained,freeze_at)
        self.useFPN = useFPN
        if self.useFPN:
            self.out_channels = out_channels

            # return_layers = {'res2': 0, 'res3': 1, 'res4': 2, 'res5': 3}
            in_channels_stage2 = self.backbone.out_channels // 8
            in_channels_dict = {
                "res2":in_channels_stage2,
                "res3":in_channels_stage2 * 2,
                "res4":in_channels_stage2 * 4,
                "res5":in_channels_stage2 * 8,
            }

            self.fpn = FPNNetV2(in_channels_dict,out_channels)
        else:
            self.out_channels = self.backbone.out_channels

    def forward(self,x):
        features = self.backbone(x)
        if self.useFPN:
            features = self.fpn(features)

        return OrderedDict(features)

class ResnetFpn(nn.Module):
    def __init__(self,model_name,pretrained=False,nofreeze_at=["res2","res3","res4","res5"],
                 out_channels=256,useFPN=False):
        super(ResnetFpn,self).__init__()
        self.backbone = Backbone(model_name,pretrained,nofreeze_at)
        self.useFPN = useFPN
        if self.useFPN:
            self.out_channels = out_channels

            # return_layers = {'res2': 0, 'res3': 1, 'res4': 2, 'res5': 3}
            in_channels_stage2 = self.backbone.out_channels // 8
            in_channels_list = [
                in_channels_stage2,
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
                in_channels_stage2 * 8,
            ]

            self.fpn = FPNNet(in_channels_list,out_channels)
        else:
            self.out_channels = self.backbone.out_channels

    def forward(self,x):
        features = self.backbone(x)
        if self.useFPN:
            features = self.fpn(features)
        else:
            features = {0:features["res5"]}
        return OrderedDict(features)

class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        # _initParmas(self, self.modules())
        _initParmasV2(self, self.modules())
        # _initParmas2(self,self.named_parameters())

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class RPNHeadV2(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors,rpn_names):
        super().__init__()
        self.names = rpn_names
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        # _initParmas(self, self.modules())
        _initParmasV2(self, self.modules())
        # _initParmas2(self,self.named_parameters())


    def forward(self, x):
        logits = {}
        bbox_reg = {}
        for name in self.names:
            t = F.relu(self.conv(x[name]))
            logits[name] = self.cls_logits(t)
            bbox_reg[name] = self.bbox_pred(t)

        return logits, bbox_reg


# 每个独立分支 不共享权重
class RPNHeadV3(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors,rpn_names):
        super().__init__()

        self.conv = nn.ModuleDict()
        self.cls_logits = nn.ModuleDict()
        self.bbox_pred = nn.ModuleDict()
        self.names = rpn_names
        for name in rpn_names:
            self.conv[name] = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            self.cls_logits[name] = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
            self.bbox_pred[name] = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        # _initParmas(self, self.modules())
        _initParmasV2(self, self.modules())
        # _initParmas2(self,self.named_parameters())

    def forward(self, x):
        logits = {}
        bbox_reg = {}
        for name in self.names:
            t = F.relu(self.conv[name](x[name]))
            logits[name] = self.cls_logits[name](t)
            bbox_reg[name] = self.bbox_pred[name](t)

        return logits, bbox_reg

class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas

class FastRCNNPredictorV2(nn.Module):
    def __init__(self,in_channels, num_classes,num_layers=2):
        super().__init__()
        # self.num_layers = num_layers
        self.conv = nn.Sequential(*[nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                                                  nn.ReLU(inplace=True)) for _ in range(num_layers)])

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        #     # nn.BatchNorm2d(in_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        #     # nn.BatchNorm2d(in_channels),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d(1)
        # )
        self.cls_score = nn.Conv2d(in_channels, num_classes, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_classes * 4, 1)

        # _initParmas(self, self.modules())
        _initParmasV2(self, self.modules())
        # _initParmas2(self,self.named_parameters())

    def forward(self,x):
        x = self.conv(x)
        x = self.avgpool(x)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores.flatten(1), bbox_deltas.flatten(1)

class RFCNPredictor(nn.Module):
    def __init__(self,in_channels,num_classes,k=7,hide_size=None,num_layers=0):
        super().__init__()
        if hide_size is None:hide_size = in_channels//2
        # self.conv = nn.Conv2d(in_channels,hide_size,1,1)
        # self.num_layers = num_layers
        # if num_layers>0:
        #     self.conv2 = nn.Sequential(*[nn.Sequential(nn.Conv2d(hide_size, hide_size, 3, 1, 1),
        #                                               nn.ReLU(inplace=True)) for _ in range(num_layers)])

        self.layer5 = _make_detnet_layer(self, in_channels=in_channels,hide_size=hide_size)
        self.cls_score = nn.Conv2d(hide_size, num_classes*k*k, 1)
        self.bbox_pred = nn.Conv2d(hide_size, num_classes*4*k*k, 1)

        # _initParmas(self, self.modules())
        _initParmasV2(self, self.modules())
        # _initParmas2(self,self.named_parameters())

    def forward(self, x):
        # x = self.conv(x)
        # if self.num_layers>0:
        #     x = self.conv2(x)

        x = self.layer5(x)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas

class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        """
        Arguments:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
        """
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d["mask_fcn{}".format(layer_idx)] = misc_nn_ops.Conv2d(
                next_feature, layer_features, kernel_size=3,
                stride=1, padding=dilation, dilation=dilation)
            d["relu{}".format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        super(MaskRCNNHeads, self).__init__(d)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)

class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super(MaskRCNNPredictor, self).__init__(OrderedDict([
            ("conv5_mask", misc_nn_ops.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
            ("relu", nn.ReLU(inplace=True)),
            ("mask_fcn_logits", misc_nn_ops.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
        ]))

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)

"""输入shape [bs,m,7,7]"""
class MaskRCNNPredictorV2(nn.Module):
    def __init__(self,in_channels,dim_reduced=256,num_classes=80,num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.conv = nn.Sequential(*[nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                                                  nn.ReLU(inplace=True)) for _ in range(num_layers)])

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 2, 2, 0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        )
        # _initParmas(self, self.modules())
        _initParmasV2(self, self.modules())
        # _initParmas2(self,self.named_parameters())

        self.up_scale =2

    def forward(self,x):
        x = self.conv(x)
        x = self.deconv(x)
        # x = F.interpolate(
        #     x, scale_factor=float(self.up_scale), mode="bilinear", align_corners=False
        # )
        return x


class KeypointRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers):
        d = []
        next_feature = in_channels
        for l in layers:
            d.append(misc_nn_ops.Conv2d(next_feature, l, 3, stride=1, padding=1))
            d.append(nn.ReLU(inplace=True))
            next_feature = l
        super(KeypointRCNNHeads, self).__init__(*d)
        for m in self.children():
            if isinstance(m, misc_nn_ops.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)


class KeypointRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super(KeypointRCNNPredictor, self).__init__()
        input_features = in_channels
        deconv_kernel = 4
        self.kps_score_lowres = misc_nn_ops.ConvTranspose2d(
            input_features,
            num_keypoints,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )
        nn.init.kaiming_normal_(
            self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        self.up_scale = 2
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        x = misc_nn_ops.interpolate(
            x, scale_factor=float(self.up_scale), mode="bilinear", align_corners=False
        )
        return x


class KeypointRCNNPredictorV2(nn.Module):
    def __init__(self,in_channels,dim_reduced=256,num_keypoints=17,num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.conv = nn.Sequential(*[nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                                                  nn.ReLU(inplace=True)) for _ in range(num_layers)])

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 2, 2, 0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, in_channels, 2, 2, 0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_reduced, num_keypoints, 1, 1, 0),
            # nn.Sigmoid()
        )

        self.up_scale = 2

        # _initParmas(self, self.modules())
        _initParmasV2(self, self.modules())
        # _initParmas2(self,self.named_parameters())


    def forward(self,x):
        x = self.conv(x)
        x = self.deconv(x)
        # x = F.interpolate(
        #     x, scale_factor=float(self.up_scale), mode="bilinear", align_corners=False
        # )
        # x = F.interpolate(
        #     x, scale_factor=float(self.up_scale), mode="bilinear", align_corners=False
        # )
        return x

# -------------------------------------------------
from torchvision.models.resnet import BasicBlock,Bottleneck

# 增加一层
class BackboneV3(nn.Module):
    def __init__(self,model_name="resnet18",pretrained=False,freeze_at=["res1","res2","res3","res4","res5"],layers=2):# ["res1","res2","res3","res4","res5"]
        super().__init__()
        model_dict = {'resnet18': 512,
                      'resnet34': 512,
                      'resnet50': 2048,
                      'resnet101': 2048,
                      'resnet152': 2048,
                      'resnext50_32x4d': 2048,
                      'resnext101_32x8d': 2048,
                      'wide_resnet50_2': 2048,
                      'wide_resnet101_2': 2048}

        assert model_name in model_dict, "%s must be in %s" % (model_name, model_dict.keys())

        self.out_channels = model_dict[model_name]
        _model = torchvision.models.resnet.__dict__[model_name](pretrained=pretrained)
        # backbone_size = _model.inplanes

        layer0 = nn.Sequential(
            _model.conv1,
            _model.bn1,
            _model.relu,
            _model.maxpool
        )
        self.layer5 = _model._make_layer(Bottleneck,self.out_channels//4,layers,stride=2,dilate=False)

        for m in self.layer5.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight, std=0.01)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # if m.bias is not None:
                #     nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


        self.backbone = nn.ModuleDict(OrderedDict([ # nn.Sequential
            ("res1",layer0),         # stride = 4
            ("res2", _model.layer1), # stride = 4
            ("res3", _model.layer2), # stride = 8
            ("res4", _model.layer3), # stride = 16 --> rpn
            ("res5", _model.layer4), # stride = 32
            # ("res6",layer5)          # stride = 64 --> roi pool
        ]))




        # 参数冻结
        for name in freeze_at:
            for parameter in self.backbone[name].parameters():
                parameter.requires_grad_(False)

        # 统计所有可更新梯度的变量
        print("只有以下变量做梯度更新:")
        for name,parameter in self.backbone.named_parameters():
            if parameter.requires_grad:
                print("name:",name)

        # 默认冻结 BN中的参数 不更新
        # for m in self.backbone.modules():
        #     if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         for parameter in m.parameters():
        #             parameter.requires_grad_(False)

    def forward(self,x):
        out={}
        x = self.backbone.res1(x)
        x = self.backbone.res2(x)
        out["res2"]=x
        x = self.backbone.res3
        out["res3"] = x
        x = self.backbone.res4(x)
        out["res4"] = x
        x = self.backbone.res5(x)
        out["res5"] = x
        # x = self.backbone.res6(x)
        x = self.layer5(x)
        out["res6"] = x
        return out

# ---------------------------------------------------
class detnet_bottleneck(nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(detnet_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False,dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        # self.seblock = SeBlock(self.expansion*planes)
        self.seblock = CBAM(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes or block_type=='B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        # _initParmas(self, self.modules())
        _initParmasV2(self, self.modules())
        # _initParmas2(self,self.named_parameters())


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.seblock(out)

        out += self.downsample(x)
        out = F.relu(out)
        return out

def _make_detnet_layer(self, in_channels,hide_size=256):
    layers = []
    layers.append(detnet_bottleneck(in_planes=in_channels, planes=hide_size, block_type='B'))
    layers.append(detnet_bottleneck(in_planes=hide_size, planes=hide_size, block_type='A'))
    layers.append(detnet_bottleneck(in_planes=hide_size, planes=hide_size, block_type='A'))
    return nn.Sequential(*layers)

class RPNHeadYolov1(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super().__init__()

        self.layer5 = _make_detnet_layer(self,in_channels=in_channels)
        self.avgpool = nn.AvgPool2d(2)  # fit 448 input size
        self.conv_end = nn.Conv2d(256, num_anchors*5+num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_end = nn.BatchNorm2d(num_anchors*5+num_classes)

        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # _initParmas(self, self.modules())
        _initParmasV2(self, self.modules())
        # _initParmas2(self,self.named_parameters())


    def forward(self,features):
        x = features["res5"]
        x = self.layer5(x)
        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        x = self.conv_end(x)
        x = self.bn_end(x)
        # x = torch.sigmoid(x)
        x = x.permute(0, 2, 3, 1)  # (-1,7,7,30)
        bs = x.size(0)
        bbox_reg = x[...,:self.num_anchors*5].contiguous().view(bs,-1,self.num_anchors,5)
        logits = x[...,self.num_anchors*5:].contiguous().view(bs,-1,self.num_classes)
        return logits, bbox_reg

# ----------------retinanet-------------------------------------
class BackboneV2_retinanet(nn.Module):
    def __init__(self,model_name="resnet18",pretrained=False,freeze_at=["res1","res2","res3","res4","res5"]):# ["res1","res2","res3","res4","res5"]
        super().__init__()
        model_dict = {'resnet18': 512,
                      'resnet34': 512,
                      'resnet50': 2048,
                      'resnet101': 2048,
                      'resnet152': 2048,
                      'resnext50_32x4d': 2048,
                      'resnext101_32x8d': 2048,
                      'wide_resnet50_2': 2048,
                      'wide_resnet101_2': 2048}

        assert model_name in model_dict, "%s must be in %s" % (model_name, model_dict.keys())

        self.out_channels = model_dict[model_name]
        _model = torchvision.models.resnet.__dict__[model_name](pretrained=pretrained)
        # backbone_size = _model.inplanes

        layer0 = nn.Sequential(
            _model.conv1,
            _model.bn1,
            _model.relu,
            _model.maxpool
        )
        self.backbone = nn.ModuleDict(OrderedDict([ # nn.Sequential
            ("res1",layer0),
            ("res2", _model.layer1),
            ("res3", _model.layer2),
            ("res4", _model.layer3),
            ("res5", _model.layer4),
        ]))

        self.res6 = nn.Conv2d(self.out_channels,self.out_channels,3,2,1)
        self.res7 = nn.Conv2d(self.out_channels,self.out_channels,3,2,1)


        _initParmasV2(self, self.res6.modules())
        _initParmasV2(self, self.res7.modules())
        # _initParmas2(self,self.named_parameters())


        # 参数冻结
        for name in freeze_at:
            for parameter in self.backbone[name].parameters():
                parameter.requires_grad_(False)

        # 统计所有可更新梯度的变量
        print("只有以下变量做梯度更新:")
        for name,parameter in self.backbone.named_parameters():
            if parameter.requires_grad:
                print("name:",name)

        # 默认冻结 BN中的参数 不更新
        # for m in self.backbone.modules():
        #     if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         for parameter in m.parameters():
        #             parameter.requires_grad_(False)

    def forward(self,x):
        out={}
        x = self.backbone.res1(x)
        x = self.backbone.res2(x)
        out["res2"]=x
        x = self.backbone.res3(x)
        out["res3"] = x
        x = self.backbone.res4(x)
        out["res4"] = x
        x = self.backbone.res5(x)
        out["res5"] = x
        x = self.res6(x)
        out["res6"] = x
        x = F.relu(self.res7(x))
        out["res7"] = x

        return out

class ResnetFpnV2_retinanet(nn.Module):
    def __init__(self,model_name,pretrained=False,freeze_at=["res1","res2","res3","res4","res5"],
                 out_channels=256,useFPN=False):
        super().__init__()
        self.backbone = BackboneV2_retinanet(model_name,pretrained,freeze_at)
        self.useFPN = useFPN
        if self.useFPN:
            self.out_channels = out_channels

            # return_layers = {'res2': 0, 'res3': 1, 'res4': 2, 'res5': 3}
            in_channels_stage2 = self.backbone.out_channels // 8
            in_channels_dict = {
                "res2":in_channels_stage2,
                "res3":in_channels_stage2 * 2,
                "res4":in_channels_stage2 * 4,
                "res5":in_channels_stage2 * 8,
                "res6":in_channels_stage2 * 8,
                "res7":in_channels_stage2 * 8,
            }

            self.fpn = FPNNetV2(in_channels_dict,out_channels)
            # self.fpn = PANet(in_channels_dict,out_channels)
        else:
            self.out_channels = self.backbone.out_channels

    def forward(self,x):
        features = self.backbone(x)
        if self.useFPN:
            features = self.fpn(features)

        return OrderedDict(features)


class ResnetPANet_retinanet(nn.Module):
    def __init__(self,model_name,pretrained=False,freeze_at=["res1","res2","res3","res4","res5"],
                 out_channels=256,useFPN=False):
        super().__init__()
        self.backbone = BackboneV2_retinanet(model_name,pretrained,freeze_at)
        self.useFPN = useFPN
        if self.useFPN:
            self.out_channels = out_channels

            # return_layers = {'res2': 0, 'res3': 1, 'res4': 2, 'res5': 3}
            in_channels_stage2 = self.backbone.out_channels // 8
            in_channels_dict = {
                "res2":in_channels_stage2,
                "res3":in_channels_stage2 * 2,
                "res4":in_channels_stage2 * 4,
                "res5":in_channels_stage2 * 8,
                "res6":in_channels_stage2 * 8,
                "res7":in_channels_stage2 * 8,
            }

            # self.fpn = FPNNetV2(in_channels_dict,out_channels)
            self.fpn = PANet(in_channels_dict,out_channels)
        else:
            self.out_channels = self.backbone.out_channels

    def forward(self,x):
        features = self.backbone(x)
        if self.useFPN:
            features = self.fpn(features)

        return OrderedDict(features)

# ---------------PANet--------------------------------
class PANet(nn.Module):
    def __init__(self,in_channels_dict,out_channels=256):
        super().__init__()
        self.fpn = FPNNetV2(in_channels_dict,out_channels)

        self.conv1 = nn.ModuleDict()
        self.conv2 = nn.ModuleDict()
        for name in in_channels_dict:
            self.conv1[name] = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 2, 1),
                nn.ReLU()
            )
            self.conv2[name] = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.ReLU()
            )

        # _initParmas(self, self.modules())
        _initParmasV2(self, self.modules())
        # _initParmas2(self,self.named_parameters())

    def forward(self,features):
        features = self.fpn(features) # 先做fpn
        outs = {}
        pre_name = ""
        for i, name in enumerate(sorted(features)):  # 至下而上  [P2,P3,P4,P5]
            if i==0:
                outs[name] = features[name]
            else:
                outs[name] = self.conv2[name](self.conv1[name](outs[pre_name])+features[name])

            pre_name = name

        return outs


"""参考 PANet,输入shape [bs,m,14,14]"""
class MaskRCNNPredictorV3(nn.Module):
    def __init__(self,in_channels,dim_reduced=256,num_classes=80,num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.conv = nn.Sequential(*[nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                                                  nn.ReLU(inplace=True)) for _ in range(num_layers)])
        self.conv_fc = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,1,1),nn.ReLU(),
            nn.Conv2d(in_channels,in_channels//2,3,1,1),nn.ReLU(),
            Flatten(),
            nn.Linear(14*14*in_channels//2,28*28),
            nn.ReLU()
        )


        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        )

        # _initParmas(self, self.modules())
        _initParmasV2(self, self.modules())
        # _initParmas2(self,self.named_parameters())

        self.up_scale =2

    def forward(self,x):

        t = self.conv[:3](x)
        x1 = self.deconv(self.conv[3:](x))
        bs,c,h,w = x1.shape
        x2 = self.conv_fc(t).contiguous().view(bs,1,h,w)

        x = x1+x2

        return x

class AdaptiveFeaturePooling(nn.Module):
    def __init__(self,names,strides,useAlign=False,op="max"):
        super().__init__()
        self.names = names
        self.strides = strides
        self.useAlign = useAlign
        self.op = op
        assert op in ["max","sum","mean"]

    def forward(self,features,propose):
        """
        传统的FPN做法是按提议框的大小分级，映射到对应级的特征上，即每个提议框只会映射到一个特征map
        PAnet 是将propose 映射到所有特征级别 (每个特征都做映射)，再取 max or sum or mean"""
        out = []
        for name,stride in zip(self.names,self.strides):
            if self.useAlign:
                x = roi_pool(features[name], propose, 14, spatial_scale=1.0 / stride)
            else:
                x = roi_align(features[name], propose, 14, spatial_scale=1.0 / stride, aligned=False)

            out.append(x)

        x = torch.stack(out,-1)

        if self.op=="mean":
            x = x.mean(-1)
        elif self.op =="sum":
            x = x.sum(-1)
        else:
            x = x.max(-1)

        return x


# -----------------------dla net------------------------------------------------
class Backbone_dla_s32(nn.Module):
    def __init__(self,model_name="dla34",pretrained=False,freeze_at=["res1","res2","res3","res4","res5"]):
        super().__init__()

        _model = dlav0.__dict__[model_name](pretrained=pretrained)
        self.out_channels = _model.channels[-1]

        self.backbone = nn.ModuleDict(OrderedDict([ # nn.Sequential
            ("res1", nn.Sequential(_model.base_layer,_model.level0,_model.level1)),
            ("res2", _model.level2),
            ("res3", _model.level3),
            ("res4", _model.level4),
            ("res5", _model.level5)
        ]))


        # 参数冻结
        for name in freeze_at:
            for parameter in self.backbone[name].parameters():
                parameter.requires_grad_(False)

        # 统计所有可更新梯度的变量
        print("只有以下变量做梯度更新:")
        for name,parameter in self.backbone.named_parameters():
            if parameter.requires_grad:
                print("name:",name)

        # 默认冻结 BN中的参数 不更新
        # for m in self.backbone.modules():
        #     if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         for parameter in m.parameters():
        #             parameter.requires_grad_(False)

    def forward(self,x):
        out={}
        x = self.backbone.res1(x)
        x = self.backbone.res2(x)
        out["res2"]=x
        x = self.backbone.res3(x)
        out["res3"] = x
        x = self.backbone.res4(x)
        out["res4"] = x
        x = self.backbone.res5(x)
        out["res5"] = x

        return out

class Backbone_dla_s16(nn.Module):
    def __init__(self,model_name="dla34",pretrained=False,freeze_at=["res1","res2","res3","res4","res5"]):
        super().__init__()

        _model = dlav1.__dict__[model_name](pretrained=pretrained)
        self.out_channels = _model.channels[-1]

        self.backbone = nn.ModuleDict(OrderedDict([ # nn.Sequential
            ("res1", nn.Sequential(_model.base_layer,_model.level0,_model.level1)),
            ("res2", _model.level2),
            ("res3", _model.level3),
            ("res4", _model.level4),
            ("res5", _model.level5)
        ]))


        # 参数冻结
        for name in freeze_at:
            for parameter in self.backbone[name].parameters():
                parameter.requires_grad_(False)

        # 统计所有可更新梯度的变量
        print("只有以下变量做梯度更新:")
        for name,parameter in self.backbone.named_parameters():
            if parameter.requires_grad:
                print("name:",name)

        # 默认冻结 BN中的参数 不更新
        # for m in self.backbone.modules():
        #     if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         for parameter in m.parameters():
        #             parameter.requires_grad_(False)

    def forward(self,x):
        out={}
        x = self.backbone.res1(x)
        x = self.backbone.res2(x)
        out["res2"]=x
        x = self.backbone.res3(x)
        out["res3"] = x
        x = self.backbone.res4(x)
        out["res4"] = x
        x = self.backbone.res5(x)
        out["res5"] = x

        return out


if __name__=="__main__":
    # x = torch.rand([1,256,7,7])
    # net = ResnetFpnV2("resnet18",useFPN=True)
    # net = KeypointRCNNPredictorV2(256,256,17,2)
    x = torch.randn([1,3,224,224])
    # net = BackboneV2_retinanet("resnet18",True)
    # net = ResnetFpnV2_retinanet("resnet18",useFPN=True)
    # net = RPNHeadYolov1(256,21,3)
    # net = RPNHeadV3(256,3,["res3","res4"])
    # net = MaskRCNNPredictorV2(256)
    # x = torch.randn([1, 256, 14, 14])
    # net = MaskRCNNPredictorV3(256)

    net = Backbone_dla_s16()

    # print(net)
    pred = net(x)
    print(pred)
