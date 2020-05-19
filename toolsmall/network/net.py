from torchvision.ops import misc as misc_nn_ops
from torchvision.models import resnet
from torch import nn
from collections import OrderedDict
import torchvision
import torch
from torch.nn import functional as F

__all__=["Backbone","ResnetFpn","RPNHead","TwoMLPHead","FastRCNNPredictor"]

class Backbone(nn.Module):
    def __init__(self,model_name,pretrained=False,nofreeze_at=["res2","res3","res4","res5"]):# ["res1","res2","res3","res4","res5"]
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

if __name__=="__main__":
    x = torch.rand([1,3,96,96])
    net = ResnetFpn("resnet18",useFPN=False)
    pred = net(x)
    print()
