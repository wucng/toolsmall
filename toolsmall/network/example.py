from torch import nn
import torch
from torch.nn import functional as F
# from math import log

try:
    from .net import Backbone_dla_s16, Backbone_s16, _initParmas, _make_detnet_layer, \
        BackboneV2, Backbone_dla_s32, ResnetFpnV2_retinanet
    from .network import FPNNet_BN, PANet_DTU
except:
    from toolsmall.network.net import Backbone_dla_s16,Backbone_s16,_initParmas,_make_detnet_layer,\
        BackboneV2,Backbone_dla_s32,ResnetFpnV2_retinanet
    from toolsmall.network.network import FPNNet_BN,PANet_DTU

class SimpleNet(nn.Module):
    """不使用FPN 和 PAN"""
    def __init__(self,
                 # backbone
                 BackBoneNet:nn.Module=None,
                 model_name="resnet18", pretrained=False,
                 freeze_at=["res1","res2", "res3", "res4", "res5"],
                 out_channels=256,
                 # rpn head
                 RPNNet:nn.Module=None,
                 num_anchors=6,
                 num_classes=21,  # 包括背景
                 dropRate=0.5,
                 out_features_name="res5"
                 ):

        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.out_features_name = out_features_name

        if BackBoneNet is not None:
            self.backbone = BackBoneNet
        else:
            if "resnet" in model_name:
                self.backbone = Backbone_s16(model_name, pretrained, freeze_at)  # stride = 16
            elif "dla" in model_name:
                self.backbone = Backbone_dla_s16(model_name, pretrained, freeze_at)  # stride = 16
            else:
                raise ("false!")

        in_channels = self.backbone.out_channels

        if RPNNet is not None:
            self.rpn = RPNNet
        else:
            self.rpn = nn.Sequential(
                nn.Dropout(dropRate),
                _make_detnet_layer(self,in_channels, out_channels),
                nn.Conv2d(out_channels, num_anchors * (num_classes + 4), 3, 1, 1)
            )

        _initParmas(self, self.rpn.modules())

    def forward(self, x):
        features = self.backbone(x)
        x = self.rpn(features[self.out_features_name])
        # bs, c, h, w = x.shape
        # x = x.permute(0, 2, 3, 1).contiguous().view(bs, h,w,self.num_anchors,self.num_classes + 4)
        # x = x.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes + 4)
        return x

class BaseNet(nn.Module):
    """包含了SimpleNet"""
    def __init__(self,
                 # backbone
                 BackBoneNet:nn.Module=None,
                 model_name="resnet18", pretrained=False,
                 freeze_at=["res1","res2", "res3", "res4", "res5"],
                 out_channels=256,
                 # FPN
                 FPNNet:nn.ModuleList=None,
                 useFPN=False,
                 # PAN
                 PANNet:nn.ModuleList=None,
                 usePAN=False,
                 nums_FPN_PAN=1,useShare=True,
                 # rpn head
                 RPNNet:nn.Module=None,
                 num_anchors=6,
                 num_classes=21,  # 包括背景
                 dropRate=0.5,
                 out_features_name="res4"
                 ):

        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.useFPN = useFPN
        self.usePAN = usePAN
        self.nums_FPN_PAN = nums_FPN_PAN
        self.out_features_name = out_features_name
        self.useShare = useShare

        if BackBoneNet is not None:
            self.backbone = BackBoneNet
        else:
            if "resnet" in model_name:
                self.backbone = BackboneV2(model_name, pretrained, freeze_at)  # stride = 32
            elif "dla" in model_name:
                self.backbone = Backbone_dla_s32(model_name, pretrained, freeze_at)  # stride = 32
            else:
                raise ("false!")

        in_channels = self.backbone.out_channels

        if useFPN:
            self.res6 = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
            self.res7 = nn.Conv2d(in_channels, in_channels, 3, 2, 1)

        if useShare:
            _nums_FPN_PAN = nums_FPN_PAN if nums_FPN_PAN <= 2 else 2
        else:
            _nums_FPN_PAN = nums_FPN_PAN
        if useFPN:
            if FPNNet is not None:
                self.fpn = FPNNet
            else:
                self.fpn = nn.ModuleList()
                for i in range(_nums_FPN_PAN):
                    if i == 0:
                        self.fpn.append(FPNNet_BN({
                            "res7": in_channels,
                            "res6": in_channels,
                            "res5": in_channels,
                            "res4": in_channels // 2,
                            "res3": in_channels // 4,
                            "res2": in_channels // 8}, out_channels))
                    else:
                        self.fpn.append(FPNNet_BN({
                            "res7": out_channels,
                            "res6": out_channels,
                            "res5": out_channels,
                            "res4": out_channels,
                            "res3": out_channels,
                            "res2": out_channels}, out_channels))


        if usePAN and useFPN:
            if PANNet is not None:
                self.pan = PANNet
            else:
                self.pan = nn.ModuleList()
                for i in range(_nums_FPN_PAN):
                    self.pan.append(PANet_DTU({
                                      "res7":out_channels,
                                      "res6":out_channels,
                                      "res5": out_channels,
                                      "res4": out_channels,
                                      "res3": out_channels,
                                      "res2": out_channels}, out_channels,"add"))

        if RPNNet is not None:
            self.rpn = RPNNet
        else:
            self.rpn = nn.Sequential(
                nn.Dropout(dropRate),
                _make_detnet_layer(self, out_channels if useFPN else in_channels, out_channels),
                nn.Conv2d(out_channels, num_anchors * (num_classes + 4), 3, 1, 1)
            )

        if useFPN:
            _initParmas(self, self.res6.modules())
            _initParmas(self, self.res7.modules())
            _initParmas(self, self.fpn.modules())
            if usePAN:_initParmas(self, self.pan.modules())
        _initParmas(self, self.rpn.modules())

    def forward(self,x):
        features = self.backbone(x)
        if self.useFPN:
            features["res6"] = self.res6(features["res5"])
            features["res7"] = self.res7(F.relu(features["res6"]))

        for i in range(self.nums_FPN_PAN):
            if self.useFPN:
                if self.useShare:
                    features = self.fpn[i if i < 2 else 1](features)
                else:
                    features = self.fpn[i](features)
                if self.usePAN:
                    if self.useShare:
                        features = self.pan[i if i < 2 else 1](features)
                    else:
                        features = self.pan[i](features)

        x = self.rpn(features[self.out_features_name])
        # bs, c, h, w = x.shape
        # x = x.permute(0, 2, 3, 1).contiguous().view(bs, h,w,self.num_anchors,self.num_classes + 4)
        # x = x.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes + 4)
        return x

class BaseBackBone(nn.Module):
    def __init__(self,
                 # backbone
                 BackBoneNet:nn.Module=None,
                 model_name="resnet18", pretrained=False,
                 freeze_at=["res1","res2", "res3", "res4", "res5"],
                 out_channels=256,
                 # FPN
                 FPNNet:nn.ModuleList=None,
                 useFPN=False,
                 # PAN
                 PANNet:nn.ModuleList=None,
                 usePAN=False,
                 nums_FPN_PAN=1,useShare=True
                 ):

        super().__init__()
        # self.num_classes = num_classes
        # self.num_anchors = num_anchors
        self.useFPN = useFPN
        self.usePAN = usePAN
        self.nums_FPN_PAN = nums_FPN_PAN
        # self.out_features_name = out_features_name
        self.useShare = useShare

        if BackBoneNet is not None:
            self.backbone = BackBoneNet
        else:
            if "resnet" in model_name:
                self.backbone = BackboneV2(model_name, pretrained, freeze_at)  # stride = 32
            elif "dla" in model_name:
                self.backbone = Backbone_dla_s32(model_name, pretrained, freeze_at)  # stride = 32
            else:
                raise ("false!")

        in_channels = self.backbone.out_channels

        if useFPN:
            self.res6 = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
            self.res7 = nn.Conv2d(in_channels, in_channels, 3, 2, 1)

        if useShare:
            _nums_FPN_PAN = nums_FPN_PAN if nums_FPN_PAN <= 2 else 2
        else:
            _nums_FPN_PAN = nums_FPN_PAN
        if useFPN:
            if FPNNet is not None:
                self.fpn = FPNNet
            else:
                self.fpn = nn.ModuleList()
                for i in range(_nums_FPN_PAN):
                    if i == 0:
                        self.fpn.append(FPNNet_BN({
                            "res7": in_channels,
                            "res6": in_channels,
                            "res5": in_channels,
                            "res4": in_channels // 2,
                            "res3": in_channels // 4,
                            "res2": in_channels // 8}, out_channels))
                    else:
                        self.fpn.append(FPNNet_BN({
                            "res7": out_channels,
                            "res6": out_channels,
                            "res5": out_channels,
                            "res4": out_channels,
                            "res3": out_channels,
                            "res2": out_channels}, out_channels))


        if usePAN and useFPN:
            if PANNet is not None:
                self.pan = PANNet
            else:
                self.pan = nn.ModuleList()
                for i in range(_nums_FPN_PAN):
                    self.pan.append(PANet_DTU({
                                      "res7":out_channels,
                                      "res6":out_channels,
                                      "res5": out_channels,
                                      "res4": out_channels,
                                      "res3": out_channels,
                                      "res2": out_channels}, out_channels,"add"))

        if useFPN:
            _initParmas(self, self.res6.modules())
            _initParmas(self, self.res7.modules())
            _initParmas(self, self.fpn.modules())
            if usePAN:_initParmas(self, self.pan.modules())

    def forward(self,x):
        features = self.backbone(x)
        if self.useFPN:
            features["res6"] = self.res6(features["res5"])
            features["res7"] = self.res7(F.relu(features["res6"]))

        for i in range(self.nums_FPN_PAN):
            if self.useFPN:
                if self.useShare:
                    features = self.fpn[i if i < 2 else 1](features)
                else:
                    features = self.fpn[i](features)
                if self.usePAN:
                    if self.useShare:
                        features = self.pan[i if i < 2 else 1](features)
                    else:
                        features = self.pan[i](features)

        # x = self.rpn(features[self.out_features_name])
        # bs, c, h, w = x.shape
        # x = x.permute(0, 2, 3, 1).contiguous().view(bs, h,w,self.num_anchors,self.num_classes + 4)
        # x = x.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes + 4)
        return features

if __name__ == "__main__":
    x = torch.rand([1,3,256,256])
    # model = BaseNet(useFPN=True,usePAN=True,nums_FPN_PAN=3,useShare=True)
    # model = SimpleNet(BackBoneNet=ResnetFpnV2_retinanet("resnet18",useFPN=True))
    model = BaseBackBone()
    print(model)
    pred = model(x)
    # print(pred.shape)