import torch
from torch import nn
from torch.nn import functional as F

# from torchvision.ops import roi_align,roi_pool,RoIPool,RoIAlign,MultiScaleRoIAlign,PSRoIPool,PSRoIAlign,DeformConv2d
from torchvision.ops import RoIPool,RoIAlign,PSRoIPool,PSRoIAlign
from torchvision.models.resnet import resnet50,resnet34
from .dlav0 import dla34
from .moudle import _make_detnet_layer,SPP


# ---------------------------------------------------------------------------------
def _initParmas(modules):
    for m in modules:
        if isinstance(m, (nn.Conv2d,nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, std=0.01)
            # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                # nn.init.zeros_(m.bias)
                nn.init.constant_(m.bias, 0)

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

class RegionProposeNetwork(nn.Module):
    """rpn"""
    def __init__(self,in_channel,hide_channel=256,num_anchors=9,num_classes=21):
        super().__init__()
        self.rpn = nn.Sequential(
                nn.Dropout(),
                _make_detnet_layer(in_channel, hide_channel),
                nn.Conv2d(hide_channel, num_anchors * (num_classes + 4), 3, 1, 1)
            )

    def forward(self,x):
        output = self.rpn(x)

        return output

class RegionProposeNetworkV2(nn.Module):
    """rpn"""
    def __init__(self,in_channel,hide_channel=256,num_anchors=9,num_classes=21):
        super().__init__()
        self.rpn = nn.Sequential(
                nn.Dropout(),
                _make_detnet_layer(in_channel, hide_channel),
                nn.Conv2d(hide_channel,hide_channel,3,1,1),
                nn.ReLU(inplace=True),

                nn.Conv2d(hide_channel, num_anchors * num_classes, 1),
                nn.Conv2d(hide_channel, num_anchors * 4, 1),
            )

    def forward(self,x):
        x = self.rpn[:4](x)
        cls = self.rpn[4](x)
        reg = self.rpn[5](x)
        return cls,reg

class RegionProposeNetworkV3(nn.Module):
    """rpn"""
    def __init__(self,in_channel,num_anchors,num_classes=1):
        super().__init__()
        self.rpn = nn.Sequential(
            nn.Conv2d(in_channel,in_channel,3,1,1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channel,num_anchors*num_classes,1),  # Lcls
            nn.Conv2d(in_channel,num_anchors*4,1) # Lreg
        )

    def forward(self,x):
        x = self.rpn[:2](x)

        cls = self.rpn[2](x)
        reg = self.rpn[3](x)

        # bs = cls.size(0)
        # cls = cls.permute(0,2,3,1).contiguous().view(bs,-1)
        # reg = reg.permute(0,2,3,1).contiguous().view(bs,-1,4)

        return cls,reg

class RegionProposeNetworkV4(nn.Module):
    """rpn"""
    def __init__(self,in_channel,num_anchors,num_classes=1):
        super().__init__()
        self.rpn = nn.Sequential(
            SPP(in_channel,in_channel),
            nn.Conv2d(in_channel,in_channel,3,1,1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channel,num_anchors*(num_classes+4),1)
        )

    def forward(self,x):
        x = self.rpn[:3](x)

        return self.rpn[3](x)


class TwoMLPHead(nn.Module):
    def __init__(self,in_channel,useFC=False):
        super().__init__()
        if useFC:
            self.classifier = nn.Sequential(
                Flatten(),
                nn.Linear(in_channel * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                # nn.Linear(4096, num_classes),
            )
        else:
            # 使用1x1 卷积代替 FC层
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channel,512,1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),

                nn.Conv2d(512,512,7,1,0),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),

                nn.Conv2d(512, 4096, 1),
                nn.BatchNorm2d(4096),
                nn.ReLU(inplace=True),
                Flatten()
            )


    def forward(self,x):
        return self.classifier(x)

class RCNNHead(nn.Module):
    def __init__(self,in_channel,stride=16,num_classes=21,use_RoIAlign=False,roi_out_size=7,
                 twoMLPHead=None,useFC=False):
        super().__init__()
        if use_RoIAlign:
            self.roipooling = RoIAlign(roi_out_size, 1.0 / stride, -1, aligned=False)
        else:
            self.roipooling = RoIPool(roi_out_size, 1.0 / stride)

        if twoMLPHead is None:
            self.twoMLPHead = TwoMLPHead(in_channel,useFC)
        else:
            self.twoMLPHead = twoMLPHead

        self.rcnnHead_cls = nn.Linear(4096, num_classes)
        self.rcnnHead_reg = nn.Linear(4096, num_classes * 4)

    def forward(self,feature,propose):
        x = self.roipooling(feature,propose)
        x = self.twoMLPHead(x)
        scores = self.rcnnHead_cls(x)
        bbox_deltas = self.rcnnHead_reg(x)
        return scores,bbox_deltas


class TwoMLPHead_rfcn(nn.Module):
    def __init__(self,in_channel,ksize=7,num_classes=21):
        super().__init__()

        self.hideLayer = nn.Sequential(
            nn.Conv2d(in_channel, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 4096, 1),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
        )

        self.rcnnHead_cls = nn.Sequential(
            nn.Conv2d(4096,ksize*ksize*num_classes,1),
            # nn.ReLU(inplace=True)
        )
        self.rcnnHead_reg = nn.Sequential(
            nn.Conv2d(4096, ksize * ksize * num_classes*4, 1),
            # nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.hideLayer(x)
        return self.rcnnHead_cls(x),self.rcnnHead_reg(x)

class RFCNHead(nn.Module):
    def __init__(self,in_channel,stride=16,ksize=7,num_classes=21,use_RoIAlign=False,roi_out_size=7):
        super().__init__()
        self.ksize = ksize
        self.roi_out_size = roi_out_size
        self.num_classes = num_classes
        self.twoMLPHead = TwoMLPHead_rfcn(in_channel,ksize,num_classes)

        if use_RoIAlign:
            # self.roipooling = RoIAlign(roi_out_size,1.0/stride,-1,aligned=False)
            self.psroipooling = PSRoIAlign(roi_out_size, 1.0 / stride, -1)
        else:
            # self.roipooling = RoIPool(roi_out_size,1.0/stride)
            self.psroipooling = PSRoIPool(roi_out_size, 1.0 / stride)

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten()
        )

    def forward(self,feature,propose):
        rcnnHead_cls,rcnnHead_reg = self.twoMLPHead(feature)

        """
        scores = self.roipooling(rcnnHead_cls,propose)
        bbox_deltas = self.roipooling(rcnnHead_reg,propose)
        scores = F.max_pool2d(scores,self.roi_out_size,1).contiguous().view(-1,self.num_classes,self.ksize,self.ksize)
        bbox_deltas = F.max_pool2d(bbox_deltas,self.roi_out_size,1).contiguous().view(-1,self.num_classes* 4,self.ksize,self.ksize)
        """
        scores = self.psroipooling(rcnnHead_cls, propose)
        bbox_deltas = self.psroipooling(rcnnHead_reg, propose)
        # """
        scores = self.avgpool(scores)
        bbox_deltas = self.avgpool(bbox_deltas)

        return scores, bbox_deltas

class Backbone(nn.Module):
    def __init__(self,model_name="dla34",pretrained=False,more_branch=False):
        super().__init__()
        self.more_branch = more_branch
        if model_name == "dla34":
            m = dla34(pretrained)
            if not more_branch:
                # 从2->1 （网络总体stride 32 --> 16）
                m.level5.tree1.conv1.stride = (1, 1)
                m.level5.downsample = None
                # m.level5.downsample.kernel_size = (1,1)
                # m.level5.downsample.stride = (1,1)

            self.backbone = nn.Sequential(
                nn.Sequential(m.base_layer,m.level0,m.level1),
                m.level2, # 4
                m.level3, # 8
                m.level4, # 16
                m.level5  # 32->16
            )
        else:
            if model_name == "resnet34":
                m = resnet34(pretrained)
            elif model_name == "resnet50":
                m = resnet50(pretrained)

            if not more_branch:
                # 修改conv4_block1的步距，从2->1 （网络总体stride 32 --> 16）
                m.layer4[0].conv1.stride = (1, 1)
                m.layer4[0].conv2.stride = (1, 1)
                m.layer4[0].downsample[0].stride = (1, 1)

            self.backbone = nn.Sequential(
                nn.Sequential(m.conv1,m.bn1,m.relu,m.maxpool),
                m.layer1, # 4
                m.layer2, # 8
                m.layer3, # 16
                m.layer4  # 32->16
            )

    def forward(self,x):
        if self.more_branch:
            x1 = self.backbone[:2](x)
            x2 = self.backbone[2](x1)
            x3 = self.backbone[3](x2)
            x4 = self.backbone[4](x3)

            return {"res1":x1,"res2":x2,"res3":x3,"res4":x4}

        return self.backbone(x)


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
