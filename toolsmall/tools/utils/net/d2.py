from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.engine import DefaultTrainer
from detectron2.modeling import build_model
from detectron2.config import get_cfg
import torch
from torch import nn
import numpy as np

# ------------------backbone--------------------------------------

def get_model(config_file,weights=None,freeze_at=2):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.WEIGHTS = weights
    if weights is None:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

    cfg.MODEL.BACKBONE.FREEZE_AT = freeze_at

    model = build_model(cfg)
    # model = DefaultTrainer.build_model(cfg)

    # print(model)
    return model.backbone

class D2Backbone(nn.Module):
    def __init__(self,config_file,weights=None,freeze_at=5):
        super().__init__()
        self.m = get_model(config_file,weights,freeze_at)
        # tmp = ['stem','res2','res3','res4','res5']
        # # freeze layer
        # _tmp = tmp[:freeze_at]
        # for name,parameter in self.m.named_parameters():
        #     # parameter.requires_grad_(True)
        #     for _name in _tmp:
        #         if _name in name:
        #             parameter.requires_grad_(False)
        #             break


    def forward(self,x):
        """
        # p2:[1,256,56,56]
        # p3:[1,256,28,28]
        # p4:[1,256,14,14]
        # p5:[1,256,7,7]
        # p6:[1,256,4,4]
        """
        pred = self.m(x)

        return pred



from detectron2.modeling.backbone.fpn import FPN,LastLevelMaxPool
from detectron2.modeling.backbone.backbone import Backbone
# from torchvision.models.resnet import resnet34
from torchvision.models import resnet
# from torch import nn
from collections import OrderedDict

class ResnetBone(Backbone):
    def __init__(self,model_name="resnet34",pretrained=True,freeze_at=2):
        super().__init__()
        # _model = torchvision.models.resnet.__dict__[model_name](pretrained=pretrained)
        _model = resnet.__dict__[model_name](pretrained=pretrained)
        self.bottom_up = nn.Sequential(OrderedDict([
            ("stem",nn.Sequential(_model.conv1,_model.bn1,_model.relu,_model.maxpool)),
            ("res2", _model.layer1),
            ("res3", _model.layer2),
            ("res4", _model.layer3),
            ("res5", _model.layer4)
        ]))

        # freeze
        for parameter in self.bottom_up[:freeze_at].parameters():
            parameter.requires_grad_(False)

        out_channels = _model.inplanes

        self._out_features = ["res2","res3","res4","res5"]
        self._out_feature_channels = {"res2":out_channels//8,"res3":out_channels//4,"res4":out_channels//2,"res5":out_channels}
        self._out_feature_strides = {"res2":4,"res3":8,"res4":16,"res5":32}

    def forward(self,x):
        x = self.bottom_up.stem(x)
        x2 = self.bottom_up.res2(x)
        x3 = self.bottom_up.res3(x2)
        x4 = self.bottom_up.res4(x3)
        x5 = self.bottom_up.res5(x4)

        return {"res2":x2,"res3":x3,"res4":x4,"res5":x5}

def fPNBackBone(model_name="resnet34",pretrained=False,freeze_at=2):
    # ["res2", "res3", "res4", "res5"]
    in_features = ["res2","res3","res4","res5"]
    out_channels = 256
    norm = "" # "GN"
    fuse_type = "sum" # "sum" or "avg"
    backbone = FPN(
            bottom_up=ResnetBone(model_name,pretrained,freeze_at),
            in_features=in_features,
            out_channels=out_channels,
            norm=norm,
            top_block=LastLevelMaxPool(),
            fuse_type=fuse_type,
        )

    return backbone



# ------------------rpnhead--------------------------------------
class StandardRPNHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, box_dim: int = 4):
        super().__init__()
        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        t = F.relu(self.conv(x))
        pred_objectness_logits=self.objectness_logits(t)
        pred_anchor_deltas=self.anchor_deltas(t)
        return pred_objectness_logits, pred_anchor_deltas

from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm
from detectron2.layers import ConvTranspose2d, cat, interpolate
import fvcore.nn.weight_init as weight_init


class FastRCNNConvFCHead(nn.Sequential):
    def __init__(
            self, input_shape,  conv_dims, fc_dims, conv_norm=""):
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self._output_size = (input_shape[1],input_shape[2],input_shape[3])

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            if k == 0:
                self.add_module("flatten", nn.Flatten())
            fc = Linear(int(np.prod(self._output_size)), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_relu{}".format(k + 1), nn.ReLU())
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x


class BaseKeypointRCNNHead(nn.Sequential):
    """
    Implement the basic Keypoint R-CNN losses and inference logic described in
    Sec. 5 of :paper:`Mask R-CNN`.
    """

    def __init__(self,in_channels,num_keypoints,conv_dims, loss_weight=1.0, loss_normalizer=1.0):
        super().__init__()
        # default up_scale to 2.0 (this can be made an option)
        up_scale = 2.0

        for idx, layer_channels in enumerate(conv_dims, 1):
            module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1)
            self.add_module("conv_fcn{}".format(idx), module)
            self.add_module("conv_fcn_relu{}".format(idx), nn.ReLU())
            in_channels = layer_channels

        deconv_kernel = 4
        self.score_lowres = ConvTranspose2d(
            in_channels, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )
        self.up_scale = up_scale

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        for layer in self:
            x = layer(x)
        x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x


class BaseMaskRCNNHead(nn.Sequential):
    def __init__(self, cur_channels, num_classes, conv_dims, conv_norm=""):
        super().__init__()
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        self.deconv = ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        self.add_module("deconv_relu", nn.ReLU())
        cur_channels = conv_dims[-1]

        self.predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self,x):
        return self.layers(x)

    def layers(self, x):
        for layer in self:
            x = layer(x)
        return x

if __name__=="__main__":
    # x = torch.rand([1, 3, 224, 224])
    # config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    # weights = "/media/wucong/225A6D42D4FA828F1/models/detecton2/model_final_f10217.pkl"
    # # m = get_model(config_file,weights)
    # # pred = m(x) # p2:[1,256,56,56],p3:[1,256,28,28],p4:[1,256,14,14],p5:[1,256,7,7],p6:[1,256,4,4]
    # # print()
    # m = D2Backbone(config_file,weights)
    # m(x)

    # x = torch.rand([32,512,7,7])
    # # pred = FastRCNNConvFCHeadV2(512,256,1024)(x) # [32,1024]
    # pred = FastRCNNConvFCHead(x.shape,[256,256],[512,1024])(x) # [32,1024]
    # print()

    # x = torch.rand([32,512,14,14])
    # pred = BaseKeypointRCNNHead(512,17,[256,256])(x) # [32,17,56,56]
    # print()

    x = torch.rand([32, 512, 14, 14])
    pred = BaseMaskRCNNHead(512, 80, [256, 256])(x)  # [32,80,28,28]
    print()