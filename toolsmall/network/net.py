from torchvision.ops import misc as misc_nn_ops
from torchvision.models import resnet
from torch import nn
from collections import OrderedDict
import torchvision
import torch
from torch.nn import functional as F

__all__=["Backbone","ResnetFpn","RPNHead","TwoMLPHead","FastRCNNPredictor"]

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
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self,features):
        # outNmae="" # "p"
        outs={}
        last_inner = None
        i = 0
        for i,name in enumerate(sorted(features)):
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
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self,features):
        # outNmae="" # "p"
        outs={}
        last_inner = None
        name = ""
        for i,name in enumerate(sorted(features)):
            inner_lateral = self.inner_blocks[name](features[name])
            feat_shape = inner_lateral.shape[-2:]
            if last_inner is None:
                last_inner = inner_lateral
            else:
                inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
                last_inner = inner_lateral + inner_top_down
            outs[name] = self.layer_blocks[name](last_inner)

        # 最后一个做pool 只用于提取框
        outs["pool"]=F.max_pool2d(outs[name], 1, 2, 0) # "pool"

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

        for l in self.children():
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

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

        for l in self.children():
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

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

        for l in self.children():
            for name in rpn_names:
                torch.nn.init.normal_(l[name].weight, std=0.01)
                torch.nn.init.constant_(l[name].bias, 0)

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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.conv(x)
        x = self.avgpool(x)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores.flatten(1), bbox_deltas.flatten(1)

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

class MaskRCNNPredictorV2(nn.Module):
    def __init__(self,in_channels,dim_reduced=256,num_classes=80,num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.conv = nn.Sequential(*[nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                                                  nn.ReLU(inplace=True)) for _ in range(num_layers)])

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.conv(x)
        x = self.deconv(x)

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
            nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_reduced, num_keypoints, 1, 1, 0)
        )

        self.up_scale = 2

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.conv(x)
        x = self.deconv(x)
        x = F.interpolate(
            x, scale_factor=float(self.up_scale), mode="bilinear", align_corners=False
        )
        x = F.interpolate(
            x, scale_factor=float(self.up_scale), mode="bilinear", align_corners=False
        )
        return x


if __name__=="__main__":
    x = torch.rand([1,256,7,7])
    # net = ResnetFpnV2("resnet18",useFPN=True)
    net = KeypointRCNNPredictorV2(256,256,17,2)
    print(net)
    pred = net(x)
    print()
