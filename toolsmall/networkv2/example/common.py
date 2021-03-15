from torch import nn
import torch
from torch.nn import functional as F

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

# ------------注意力模块--------------
class SEblock(nn.Module):
    """https://arxiv.org/pdf/1709.01507v4.pdf"""
    def __init__(self,in_c,hide_c,out_c):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_c,hide_c),
            nn.ReLU(),
            nn.Linear(hide_c,out_c),
            nn.Sigmoid()
        )
    def forward(self,x):
        x1 = x.mean([2,3])
        x1 = self.l1(x1)
        return x*x1[...,None,None]

class SKblock(nn.Module):
    """Sknet:https://arxiv.org/pdf/1903.06586v2.pdf"""
    def __init__(self,in_c,hide_c,out_c):
        super().__init__()
        self.l1 = CBR(in_c//2,out_c,3)
        self.l2 = CBR(in_c//2,out_c,3,dilation=2)
        self.fc = nn.Sequential(
            nn.Linear(out_c,hide_c),
            nn.ReLU())
        self.fc1 = nn.Sequential(
                        nn.Linear(hide_c,out_c),
                        nn.Softmax(1))
        self.fc2 = nn.Sequential(
            nn.Linear(hide_c, out_c),
            nn.Softmax(1))


    def forward(self,x):
        x1,x2 = torch.chunk(x,2,1)
        x1 = self.l1(x1)
        x2 = self.l2(x2)
        x3=torch.mean(x1+x2,[2,3])
        x3 = self.fc(x3)
        x1 = self.fc1(x3)[...,None,None]*x1
        x2 = self.fc2(x3)[...,None,None]*x2
        x = x1+x2
        return x

class SKblockv2(nn.Module):
    """Assemble-ResNet(2020):https://arxiv.org/pdf/2001.06268v2.pdf"""
    def __init__(self,in_c,hide_c,out_c):
        super().__init__()
        self.l1 = CBR(in_c,out_c*2,3)
        self.fc = nn.Sequential(
            nn.Linear(out_c,hide_c),
            nn.ReLU())
        self.fc1 = nn.Sequential(
                        nn.Linear(hide_c,out_c),
                        nn.Softmax(1))
        self.fc2 = nn.Sequential(
            nn.Linear(hide_c, out_c),
            nn.Softmax(1))


    def forward(self,x):
        x = self.l1(x)
        x1,x2 = torch.chunk(x,2,1)
        x3=torch.mean(x1+x2,[2,3])
        x3 = self.fc(x3)
        x1 = self.fc1(x3)[...,None,None]*x1
        x2 = self.fc2(x3)[...,None,None]*x2
        x = x1+x2
        return x

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

class SAM(nn.Module): # ChannelAttention 改进版
    """https://arxiv.org/pdf/2004.10934v1.pdf"""
    def __init__(self, in_planes):
        super(SAM, self).__init__()
        self.conv = nn.Conv2d(in_planes,in_planes,3,1,1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        return self.sigmoid(out)*x

# ------------注意力模块--------------

def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class ChannelShuffle(nn.Module):
    def __init__(self,groups=4):
        self.groups = groups
        super().__init__()
    def forward(self,x):
        return channel_shuffle(x,self.groups)


class CBR(nn.Module):
    """# 使用BN bias可以设置为False"""
    def __init__(self,in_c,out_c,ksize,stride=1,padding=None,dilation=1,groups=1,bias=False,act=None):
        super().__init__()
        if padding is None:_ksize = ksize + 2*(dilation-1)
        self.cbr = nn.Sequential(
            nn.Conv2d(in_c,out_c,ksize,stride,padding if padding is not None else _ksize//2,dilation,groups,bias),
            nn.BatchNorm2d(out_c),
        )
        if act is not None:
            self.act = act
        else:
            self.act = nn.Identity()


    def forward(self,x):
        return self.act(self.cbr(x))