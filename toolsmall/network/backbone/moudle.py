import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class ShuffleBlock(nn.Module):
    """实现特征通道打乱
        参考:ShuffleNet
    """
    def __init__(self,inputs,group=4):
        super(ShuffleBlock,self).__init__()
        assert inputs%group**2==0
        # self.inputs = inputs
        self.group = group
        self.channels = inputs//group**2

    def forward(self,x):
        result = []
        new_x = x.split(self.channels,1)
        for i in range(self.group):
            for j in range(self.group):
                result.append(new_x[i+j*self.group])

        return torch.cat(result,1)

class ShuffleBlockV1(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlockV1, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        # 维度变换之后必须要使用.contiguous()使得张量在内存连续之后才能调用view函数
        return x.view(N,g,int(C/g),H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)

class SeBlock(nn.Module):
    """SENet模块
        通道 注意力机制
    """
    def __init__(self,inputs,reduces=16):
        super(SeBlock,self).__init__()
        self.fc1 = nn.Linear(inputs,inputs//reduces)
        self.fc2 = nn.Linear(inputs//reduces,inputs)

    def forward(self,x):
        x1 = torch.mean(x,[2,3])
        x1 = self.fc1(x1)
        x1 = F.relu(x1)
        x1 = self.fc2(x1)
        # x1 = F.sigmoid(x1)
        x1 = torch.sigmoid(x1)

        return x*x1[...,None,None]

class SeBlockV2(nn.Module):
    """SENet模块
        feature map 大小 注意力机制
    """
    def __init__(self,inputs=1,reduces=4):
        super(SeBlockV2,self).__init__()
        self.conv = nn.Conv2d(1,1,3,reduces,1)
        self.deconv = nn.ConvTranspose2d(1,1,3,reduces,0,1)

    def forward(self,x):
        x1 = torch.mean(x,[1],keepdim=True)
        x1 = self.conv(x1)
        x1 = F.relu(x1)
        x1 = self.deconv(x1)
        # x1 = F.sigmoid(x1)
        x1 = torch.sigmoid(x1)

        return x*x1

class SeBlockV3(nn.Module):
    """SENet模块
            feature map 大小 和通道 注意力机制
        """

    def __init__(self, inputs=1, reduces=4):
        super(SeBlockV3, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3, reduces, 1)
        self.deconv = nn.ConvTranspose2d(1, 1, 3, reduces, 0, 1)
        self.fc1 = nn.Linear(inputs, inputs // reduces**2)
        self.fc2 = nn.Linear(inputs // reduces**2, inputs)

    def forward(self,x):
        x1 = torch.mean(x,[2,3])
        x1 = self.fc1(x1)
        x1 = F.relu(x1)
        x1 = self.fc2(x1)
        # x1 = F.sigmoid(x1)
        x1 = torch.sigmoid(x1)

        x2 = torch.mean(x, [1], keepdim=True)
        x2 = self.conv(x2)
        x2 = self.deconv(x2)
        x2 = torch.sigmoid(x2)
        return x*x1[...,None,None]*x2

# ---------------------Attention-------------------------------------------------------
"""改进的 SeBlock"""
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)*x

class SAM(nn.Module): # ChannelAttention 改进版
    def __init__(self, in_planes):
        super(SAM, self).__init__()
        self.conv = nn.Conv2d(in_planes,in_planes,3,1,1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        return self.sigmoid(out)*x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x1)
        return self.sigmoid(x1)*x

class CBAM(nn.Module):
    def __init__(self,in_planes, kernel_size=7,ratio=16):
        super().__init__()
        self.m = nn.Sequential(ChannelAttention(in_planes,ratio),SpatialAttention(kernel_size))
    def forward(self,x):
        return self.m(x)


# --------------Depthwise conv-----------------------------------------
class DwConv2d(nn.Module):
    """Depthwise conv"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(DwConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class DwConv2dV1(nn.Module):
    """from torchvision.models.mobilenet"""
    def __init__(self,in_planes, out_planes, kernel_size=3, stride=1,padding=0,bias=False):
        super(DwConv2dV1,self).__init__()

        # dw
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size, stride, padding, groups=in_planes, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu6 = nn.ReLU6(inplace=True)

        # pw-linear
        self.pointwise = nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=bias)
        # self.bn = nn.BatchNorm2d(out_planes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu6(x)
        x = self.pointwise(x)
        # x = self.bn(x)
        return x

"""不推荐 速度比DwConv2dV1慢"""
class DwConv2dV2(nn.Module):
    """from torchvision.models.mobilenet
        结合res2net 通道拆分思想
    """
    def __init__(self,in_planes, out_planes, kernel_size=3, stride=1,padding=0,scale=4):
        super(DwConv2dV2,self).__init__()

        self.scale = scale
        # dw
        self.conv1 = nn.ModuleList()
        self.each_scale_in_planes = in_planes//scale
        for i in range(scale):
            self.conv1.append(nn.Sequential(nn.Conv2d(self.each_scale_in_planes, self.each_scale_in_planes, \
                              kernel_size, stride, padding, groups=self.each_scale_in_planes, bias=False),
                                            nn.BatchNorm2d(self.each_scale_in_planes),
                                            nn.ReLU6(inplace=True)))

        # self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size, stride, padding, groups=in_planes, bias=False)
        # self.bn1 = nn.BatchNorm2d(in_planes)
        # self.relu6 = nn.ReLU6(inplace=True)

        # pw-linear
        self.pointwise = nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False)
        # self.bn = nn.BatchNorm2d(out_planes)

    def forward(self,x):
        result = []
        x_list = x.split(self.each_scale_in_planes,1)
        for i in range(self.scale):
            if i > 0:
                result.append(self.conv1[i](x_list[i])+result[i-1])
            else:
                result.append(self.conv1[i](x_list[i]))
        x = torch.cat(result, 1)

        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu6(x)
        x = self.pointwise(x)
        # x = self.bn(x)
        return x

"""结合 Dilated convolution"""
class DwConv2dV3(nn.Module):
    """from torchvision.models.mobilenet"""
    def __init__(self,in_planes, out_planes, kernel_size=3, stride=1,padding=0,dilation=2):
        super(DwConv2dV3,self).__init__()

        # dw
        padding = (kernel_size+2*(dilation-1))//2
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size, stride, padding, groups=in_planes, bias=False,dilation=dilation)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu6 = nn.ReLU6(inplace=True)

        # pw-linear
        self.pointwise = nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False)
        # self.bn = nn.BatchNorm2d(out_planes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu6(x)
        x = self.pointwise(x)
        # x = self.bn(x)
        return x


"""参数量比DwConv2dV1更少"""
class MixConv2d(nn.Module):
    """MixNet"""
    def __init__(self,in_planes, out_planes,filters=[3,5,7,9],stride=1):
        super(MixConv2d,self).__init__()
        self.group = len(filters)
        self.in_planes_each_group = in_planes//self.group
        out_planes_each_group = out_planes//self.group
        self.mlist = nn.ModuleList()
        for i in filters:
            self.mlist.append(DwConv2dV1(self.in_planes_each_group,out_planes_each_group,i,stride,i//2))
            # self.mlist.append(DwConv2dV2(self.in_planes_each_group,out_planes_each_group,i,stride,i//2))

    def forward(self,x):
        result = []
        x_list = x.split(self.in_planes_each_group,1)
        for i in range(self.group):
            result.append(self.mlist[i](x_list[i]))

        return torch.cat(result,1)

# ---------------Dilated convolution----------------------------------------
# https://www.jianshu.com/p/27e2d441e668
class DilatedConv2d(nn.Module):
    """相对于传统卷积核大小为: kernel_size+2*(dilation-1)"""
    def __init__(self,in_planes, out_planes, kernel_size=3, stride=1,padding=0,dilation=1):
        super(DilatedConv2d,self).__init__()
        padding = (kernel_size+2*(dilation-1))//2
        self.dilatedconv=nn.Conv2d(in_planes,out_planes,kernel_size,stride,padding,dilation)

    def forward(self,x):
        return self.dilatedconv(x)

# -------------------------------------------------------

class Fire(nn.Module):
    """from torchvision.models.squeezenet import"""
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class FireV1(nn.Module):
    """分2个分支"""
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(FireV1, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.squeeze_planes = squeeze_planes


        self.expand1x1 = nn.Conv2d(squeeze_planes//2, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes//2, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)


        self.expand1x1_b2 = nn.Conv2d(squeeze_planes//2, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation_b2 = nn.ReLU(inplace=True)
        self.expand3x3_b2 = nn.Conv2d(squeeze_planes//2, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation_b2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        x = x.split(self.squeeze_planes//2,1)

        x11 = self.expand1x1_activation(self.expand1x1(x[0]))
        x12 = self.expand3x3_activation(self.expand3x3(x[0]))

        x21 = self.expand1x1_activation_b2(self.expand1x1_b2(x[1]))
        x22 = self.expand3x3_activation_b2(self.expand3x3_b2(x[1]))

        return torch.cat((x11,x22),1)+torch.cat((x12,x21),1)

# ------------------------------------------------------------------------------
from torchvision.models.resnet import conv3x3,conv1x1 #,BasicBlock,Bottleneck

"""res2Net"""
class BasicBlockV1(nn.Module):
    """res2net"""
    expansion = 1

    def __init__(self, inplanes, planes,stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,scale=4):
        super(BasicBlockV1, self).__init__()
        self.scale = scale
        self.each_scale_planes = planes//scale
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.ModuleList()
        for i in range(scale):
            self.conv2.append(nn.Sequential(conv3x3(self.each_scale_planes, self.each_scale_planes),
                                              norm_layer(self.each_scale_planes)))

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        x_list = out.split(self.each_scale_planes, 1)
        result = []
        for i in range(self.scale):
            if i>0:
                result.append(self.conv2[i](x_list[i])+result[i-1])
            else:
                result.append(self.conv2[i](x_list[i]))

        out = torch.cat(result,1)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BottleneckV1(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,scale=4):
        super(BottleneckV1, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        # self.conv2 = conv3x3(width, width, stride, groups, dilation)
        # self.bn2 = norm_layer(width)
        self.scale = scale
        self.each_scale_planes = width // scale
        self.conv2 = nn.ModuleList()
        for i in range(scale):
            self.conv2.append(nn.Sequential(conv3x3(self.each_scale_planes, self.each_scale_planes, stride, groups, dilation),
                                            norm_layer(self.each_scale_planes),
                                            nn.ReLU(inplace=True)))

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        x_list = out.split(self.each_scale_planes, 1)
        result = []
        for i in range(self.scale):
            if i > 0:
                result.append(self.conv2[i](x_list[i]) + result[i - 1])
            else:
                result.append(self.conv2[i](x_list[i]))

        out = torch.cat(result, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# ---------------------------------------
"""res2Net + senNet"""
# 推荐# ----------senNet放在前面-----------------------------
class BasicBlockV2(nn.Module):
    """res2net"""
    expansion = 1

    def __init__(self, inplanes, planes,stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,scale=4):
        super(BasicBlockV2, self).__init__()
        self.scale = scale
        self.each_scale_planes = planes//scale
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.ModuleList()
        for i in range(scale):
            self.conv2.append(nn.Sequential(conv3x3(self.each_scale_planes, self.each_scale_planes),
                                              norm_layer(self.each_scale_planes)))

        self.downsample = downsample
        self.stride = stride

        self.seblock = SeBlock(planes)
        # self.seblock = SeBlockV2()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        x_list = out.split(self.each_scale_planes, 1)
        result = []
        for i in range(self.scale):
            if i>0:
                result.append(self.conv2[i](x_list[i])+result[i-1])
            else:
                result.append(self.conv2[i](x_list[i]))

        out = torch.cat(result,1)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.seblock(out)

        out += identity
        out = self.relu(out)

        return out

class BottleneckV2(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,scale=4):
        super(BottleneckV2, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        # self.conv2 = conv3x3(width, width, stride, groups, dilation)
        # self.bn2 = norm_layer(width)
        self.scale = scale
        self.each_scale_planes = width // scale
        self.conv2 = nn.ModuleList()
        for i in range(scale):
            self.conv2.append(nn.Sequential(conv3x3(self.each_scale_planes, self.each_scale_planes, stride, groups, dilation),
                                            norm_layer(self.each_scale_planes),
                                            nn.ReLU(inplace=True)))

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.seblock = SeBlock(planes * self.expansion)
        # self.seblock = SeBlockV2()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        x_list = out.split(self.each_scale_planes, 1)
        result = []
        for i in range(self.scale):
            if i > 0:
                result.append(self.conv2[i](x_list[i]) + result[i - 1])
            else:
                result.append(self.conv2[i](x_list[i]))

        out = torch.cat(result, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.seblock(out)

        out += identity
        out = self.relu(out)

        return out

# ----------senNet放在后面-----------------------------
class BasicBlockV3(nn.Module):
    expansion = 1
    def __init__(self,inplanes, planes,stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,scale=4):
        super(BasicBlockV3,self).__init__()
        self.basicBlock = BasicBlockV1(inplanes, planes,stride, downsample, groups,
                 base_width, dilation, norm_layer,scale)
        self.seBlock = SeBlock(planes)
        # self.seBlock = SeBlockV2()

    def forward(self, x):
        x = self.basicBlock(x)
        x = self.seBlock(x)
        return x

class BottleneckV3(nn.Module):
    expansion = 4
    def __init__(self,inplanes, planes,stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,scale=4):
        super(BottleneckV3,self).__init__()
        self.bottleneck = BottleneckV1(inplanes, planes,stride, downsample, groups,
                 base_width, dilation, norm_layer,scale)
        self.seBlock = SeBlock(planes * self.expansion)
        # self.seBlock = SeBlockV2()

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.seBlock(x)
        return x

# ---------------------------------------
"""res2Net + senNet + shuffleNet"""
class BasicBlockV4(nn.Module):
    """res2net"""
    expansion = 1

    def __init__(self, inplanes, planes,stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,scale=4):
        super(BasicBlockV4, self).__init__()
        self.scale = scale
        self.each_scale_planes = planes//scale
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shuffleblock = ShuffleBlockV1(scale)

        self.conv2 = nn.ModuleList()
        for i in range(scale):
            self.conv2.append(nn.Sequential(conv3x3(self.each_scale_planes, self.each_scale_planes),
                                              norm_layer(self.each_scale_planes)))

        self.downsample = downsample
        self.stride = stride

        self.seblock = SeBlock(planes)
        # self.seblock = SeBlockV2()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.shuffleblock(out)

        x_list = out.split(self.each_scale_planes, 1)
        result = []
        for i in range(self.scale):
            if i>0:
                result.append(self.conv2[i](x_list[i])+result[i-1])
            else:
                result.append(self.conv2[i](x_list[i]))

        out = torch.cat(result,1)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.seblock(out)

        out += identity
        out = self.relu(out)

        return out

class BottleneckV4(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,scale=4):
        super(BottleneckV4, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        # self.conv2 = conv3x3(width, width, stride, groups, dilation)
        # self.bn2 = norm_layer(width)
        self.shuffleblock = ShuffleBlockV1(scale)

        self.scale = scale
        self.each_scale_planes = width // scale
        self.conv2 = nn.ModuleList()
        for i in range(scale):
            self.conv2.append(nn.Sequential(conv3x3(self.each_scale_planes, self.each_scale_planes, stride, groups, dilation),
                                            norm_layer(self.each_scale_planes),
                                            nn.ReLU(inplace=True)))

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.seblock = SeBlock(planes * self.expansion)
        # self.seblock = SeBlockV2()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        out = self.shuffleblock(out)
        x_list = out.split(self.each_scale_planes, 1)
        result = []
        for i in range(self.scale):
            if i > 0:
                result.append(self.conv2[i](x_list[i]) + result[i - 1])
            else:
                result.append(self.conv2[i](x_list[i]))

        out = torch.cat(result, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.seblock(out)

        out += identity
        out = self.relu(out)

        return out


# ---------------------------------------
def conv3x3_dw(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=dilation, groups=groups, bias=False, dilation=dilation)
    # return DwConv2dV1(in_planes,out_planes,kernel_size=3,stride=stride,padding=dilation)
    # return DwConv2dV3(in_planes,out_planes,kernel_size=3,stride=stride,dilation=2)
    return MixConv2d(in_planes,out_planes,stride=stride)
    # return DilatedConv2d(in_planes,out_planes,stride=stride,dilation=2)

"""res2Net + CBAM[senNet] + shuffleNet + DwConv2d"""
class BasicBlockV5(nn.Module):
    """res2net"""
    expansion = 1

    def __init__(self, inplanes, planes,stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,scale=4):
        super(BasicBlockV5, self).__init__()
        self.scale = scale
        self.each_scale_planes = planes//scale
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3_dw(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shuffleblock = ShuffleBlockV1(scale)

        self.conv2 = nn.ModuleList()
        for i in range(scale):
            self.conv2.append(nn.Sequential(conv3x3_dw(self.each_scale_planes, self.each_scale_planes),
                                              norm_layer(self.each_scale_planes)))

        self.downsample = downsample
        self.stride = stride

        # self.seblock = SeBlock(planes)
        # self.seblock = SeBlockV2()
        self.seblock = CBAM(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.shuffleblock(out)

        x_list = out.split(self.each_scale_planes, 1)
        result = []
        for i in range(self.scale):
            if i>0:
                result.append(self.conv2[i](x_list[i])+result[i-1])
            else:
                result.append(self.conv2[i](x_list[i]))

        out = torch.cat(result,1)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.seblock(out)

        out += identity
        out = self.relu(out)

        return out

class BottleneckV5(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,scale=4):
        super(BottleneckV5, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        # self.conv2 = conv3x3(width, width, stride, groups, dilation)
        # self.bn2 = norm_layer(width)
        self.shuffleblock = ShuffleBlockV1(scale)

        self.scale = scale
        self.each_scale_planes = width // scale
        self.conv2 = nn.ModuleList()
        for i in range(scale):
            self.conv2.append(nn.Sequential(conv3x3_dw(self.each_scale_planes, self.each_scale_planes, stride, groups, dilation),
                                            norm_layer(self.each_scale_planes),
                                            nn.ReLU(inplace=True)))

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # self.seblock = SeBlock(planes * self.expansion)
        self.seblock = CBAM(planes * self.expansion)
        # self.seblock = SeBlockV2()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        out = self.shuffleblock(out)
        x_list = out.split(self.each_scale_planes, 1)
        result = []
        for i in range(self.scale):
            if i > 0:
                result.append(self.conv2[i](x_list[i]) + result[i - 1])
            else:
                result.append(self.conv2[i](x_list[i]))

        out = torch.cat(result, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.seblock(out)

        out += identity
        out = self.relu(out)

        return out


def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                    nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode="fan_out",
                                     nonlinearity="sigmoid")
            nn.init.zeros_(m.bias)

if __name__=="__main__":
    x = torch.randn([32,64,12,12]).clone()
    # m = DwConv2dV3(64,64,3,1,1)
    # m = DwConv2dV2(64,64,3,1,1)
    # m = DilatedConv2d(64,64,3,1,1,2)
    # m = MixConv2d(64,64,stride=1)
    # m = ChannelAttention(64)
    # m = SpatialAttention(3)
    m = CBAM(64,7,16)
    print(m)
    x = m(x)

    print(x.shape)
