import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_,clip_grad_value_
import numpy as np
from PIL import Image
import os

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return torch.flatten(x,1)

class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()
    def forward(self, x):
        if x.dim()==1:
            return x[None,:,None,None]
        return x[...,None,None]

class Resize(nn.Module):
    def __init__(self,in_channels,h,w):
        super(Resize, self).__init__()
        # self.in_channels = in_channels
        self.h = h
        self.w = w
        self.out = in_channels//(h*w)

    def forward(self, x):
        if x.dim()==1:
            x = x[None,:]
        return x.contiguous().view(x.size(0),self.out,self.h,self.w)


def ConvBL(in_channels=3, out_channels=32,
               kernel_size=3, stride=1, padding=1,
               bias=False,negative_slope=0.2,useBN=True):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=bias),
        # nn.BatchNorm2d(out_channels) if useBN else nn.Identity(),
        PixelwiseNormalization() if useBN else nn.Identity(),
        nn.LeakyReLU(negative_slope, inplace=True),
    )

# 参考：SAGAN
class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out#, attention


class PixelwiseNormalization(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        factor = ((x ** 2).mean(dim=1, keepdim=True) + 1e-8) ** 0.5
        return x / factor

class MinibatchStdLayer(nn.Module):

    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    # Implementation from:
    # https://github.com/facebookresearch/pytorch_GAN_zoo/blob/master/models/networks/custom_layers.py
    def forward(self, x):
        size = x.size()
        subGroupSize = min(size[0], self.group_size)
        if size[0] % subGroupSize != 0:
            subGroupSize = size[0]
        G = int(size[0] / subGroupSize)
        if subGroupSize > 1:
            y = x.view(-1, subGroupSize, size[1], size[2], size[3])
            y = torch.var(y, 1)
            y = torch.sqrt(y + 1e-8)
            y = y.view(G, -1)
            y = torch.mean(y, 1).view(G, 1)
            y = y.expand(G, size[2] * size[3]).view((G, 1, 1, size[2], size[3]))
            y = y.expand(G, subGroupSize, -1, -1, -1)
            y = y.contiguous().view((-1, 1, size[2], size[3]))
        else:
            y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)

        return torch.cat([x, y], dim=1)


# custom weights initialization called on netG and netD
def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

class BaseGAN(nn.Module):
    def __init__(self,nc=3,nz=100,dim=64,num_classes=10,condition=False):
        super().__init__()
        self.nc = nc
        self.nz = nz
        self.dim = dim
        self.num_classes = num_classes
        self.condition = condition

    def generator(self):
        raise("Manual implementation")

    def discriminator(self):
        raise("Manual implementation")

    def g_forward(self):
        return self.generator()

    def d_forward(self):
        return self.discriminator()

class Discriminator(nn.Module): # imgsize = 64
    def __init__(self, nc = 3,ndf = 64,num_classes=10,condition=False):
        super().__init__()
        self.num_classes = num_classes
        self.condition = condition # 是否使用条件GAN

        if condition:
            self._conv1_1 = nn.Sequential(
                nn.Conv2d(nc, ndf // 2, 4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self._conv1_2 = nn.Sequential(
                nn.Conv2d(num_classes, ndf // 2, 4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.d = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                            nn.LeakyReLU(0.2, inplace=True)) if not condition else nn.Identity(),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), # [4,4]->[1,1]
            Flatten(),
            # nn.Sigmoid()
        )

        # weights_init(self)

    def forward(self, x,y=None):
        if self.condition:
            bs, _, img_h, img_w = x.size()
            y = F.one_hot(y.long(), self.num_classes).float()
            y = y[..., None, None]
            y = y.expand((bs, self.num_classes, img_h, img_w))

            x = self._conv1_1(x)
            y = self._conv1_2(y)

            x = torch.cat([x, y], 1)
        x = self.d(x)
        return x

class Generator(nn.Module): # imgsize = 64
    def __init__(self,nc=3, nz = 100,ngf = 64,num_classes = 10,condition=False):
        super().__init__()
        self.num_classes = num_classes
        self.condition = condition  # 是否使用条件GAN

        if condition:
            self.deconv1_1 = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),  # [512,4,4]
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
            )

            self.deconv1_2 = nn.Sequential(
                nn.ConvTranspose2d(num_classes, ngf * 4, 4, 1, 0, bias=False),  # [512,4,4]
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
            )

        self.g = nn.Sequential(
            # input is Z, going into a convolution
            nn.Sequential(nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False), # [1,1]->[4,4]
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)) if not condition else nn.Identity(),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid(), # 对应图像 norm (0.,1.)
            # nn.Hardtanh(0, 1, True), # [0.,1.]
            # nn.Tanh() # 对应图像norm -1.0~1.0

        )

        # weights_init(self)

    def forward(self, x,y=None):
        if self.condition:
            bs, _, img_h, img_w = x.size()
            y = F.one_hot(y.long(), self.num_classes).float()
            y = y[..., None, None]
            y = y.expand((bs, self.num_classes, img_h, img_w))

            x = self.deconv1_1(x)
            y = self.deconv1_2(y)

            x = torch.cat([x, y], 1)
        return self.g(x)

class GAN(BaseGAN):
    def __init__(self,nc=3,nz=100,dim=64,num_classes=10,condition=False):
        super().__init__(nc,nz,dim,num_classes,condition)

    def generator(self):
        return Generator(self.nc,self.nz,self.dim,self.num_classes,self.condition)

    def discriminator(self):
        return Discriminator(self.nc,self.dim,self.num_classes,self.condition)

# --------------------------------------------------------------------------------
class Discriminator2(nn.Module): # imgsize = 64
    def __init__(self, nc = 3,ndf = 64,num_classes=10,condition=False):
        super().__init__()
        self.num_classes = num_classes
        self.condition = condition # 是否使用条件GAN

        if condition:
            self._conv1_1 = ConvBL(nc,ndf // 2, 3,1,1,useBN=False)

            self._conv1_2 = ConvBL(num_classes, ndf // 2, 3, 1, 1, useBN=False)

        self.d = nn.Sequential(
            # input is (nc) x 64 x 64
            ConvBL(nc, ndf, 3, 1, 1, useBN=False) if not condition else nn.Identity(),
            nn.AvgPool2d(2,2),

            ConvBL(ndf, ndf * 2, 3, 1, 1),
            nn.AvgPool2d(2, 2),

            ConvBL(ndf*2, ndf * 4, 3, 1, 1),
            nn.AvgPool2d(2, 2),

            ConvBL(ndf * 4, ndf * 8, 3, 1, 1),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), # [4,4]->[1,1]
            Flatten(),
            # nn.Sigmoid()
        )

        # weights_init(self)

    def forward(self, x,y=None):
        if self.condition:
            bs, _, img_h, img_w = x.size()
            y = F.one_hot(y.long(), self.num_classes).float()
            y = y[..., None, None]
            y = y.expand((bs, self.num_classes, img_h, img_w))

            x = self._conv1_1(x)
            y = self._conv1_2(y)

            x = torch.cat([x, y], 1)
        x = self.d(x)
        return x

class Generator2(nn.Module): # imgsize = 64
    def __init__(self,nc=3, nz = 100,ngf = 64,num_classes = 10,condition=False):
        super().__init__()
        self.num_classes = num_classes
        self.condition = condition  # 是否使用条件GAN

        if condition:
            self.deconv1_1 = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),  # [512,4,4]
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
            )

            self.deconv1_2 = nn.Sequential(
                nn.ConvTranspose2d(num_classes, ngf * 4, 4, 1, 0, bias=False),  # [512,4,4]
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
            )

        self.g = nn.Sequential(
            # input is Z, going into a convolution
            nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), # [1,1]->[4,4]
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)) if not condition else nn.Identity(),

            # state size. (ngf*8) x 4 x 4
            nn.Upsample(scale_factor=2),
            ConvBL(ngf * 8, ngf * 4, 3, 1, 1),

            # state size. (ngf*4) x 8 x 8
            nn.Upsample(scale_factor=2),
            ConvBL(ngf * 4, ngf * 2, 3, 1, 1),

            # state size. (ngf*2) x 16 x 16
            nn.Upsample(scale_factor=2),
            ConvBL(ngf * 2, ngf, 3, 1, 1),

            # state size. (ngf) x 32 x 32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Sigmoid(), # 对应图像 norm (0.,1.)
            # nn.Hardtanh(0, 1, True), # [0.,1.]
            # nn.Tanh() # 对应图像norm -1.0~1.0

        )

        # weights_init(self)

    def forward(self, x,y=None):
        if self.condition:
            bs, _, img_h, img_w = x.size()
            y = F.one_hot(y.long(), self.num_classes).float()
            y = y[..., None, None]
            y = y.expand((bs, self.num_classes, img_h, img_w))

            x = self.deconv1_1(x)
            y = self.deconv1_2(y)

            x = torch.cat([x, y], 1)
        return self.g(x)

class GAN2(BaseGAN):
    def __init__(self,nc=3,nz=100,dim=64,num_classes=10,condition=False):
        super().__init__(nc,nz,dim,num_classes,condition)

    def generator(self):
        return Generator2(self.nc,self.nz,self.dim,self.num_classes,self.condition)

    def discriminator(self):
        return Discriminator2(self.nc,self.dim,self.num_classes,self.condition)

# ---------参考：ProGAN 结构-----------------------------------------------
class Discriminator3(nn.Module): # imgsize = 64
    def __init__(self, nc = 3,ndf = 64,num_classes=10,condition=False):
        super().__init__()
        self.num_classes = num_classes
        self.condition = condition # 是否使用条件GAN

        if condition:
            self._conv1_1 = ConvBL(nc,ndf // 2, 3,1,1,useBN=False)

            self._conv1_2 = ConvBL(num_classes, ndf // 2, 3, 1, 1, useBN=False)

        self.d = nn.Sequential(
            # input is (nc) x 64 x 64
            ConvBL(nc, ndf, 3, 1, 1, useBN=False) if not condition else nn.Identity(),
            nn.AvgPool2d(2,2),

            ConvBL(ndf, ndf * 2, 3, 1, 1),
            nn.AvgPool2d(2, 2),

            ConvBL(ndf*2, ndf * 4, 3, 1, 1),
            nn.AvgPool2d(2, 2),

            ConvBL(ndf * 4, ndf * 8, 3, 1, 1),
            nn.AvgPool2d(2, 2),

            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), # [4,4]->[1,1]
            Flatten(),
            nn.Linear(ndf * 8*4*4,1,bias=False),

            # nn.Sigmoid()
        )

        # weights_init(self)

    def forward(self, x,y=None):
        if self.condition:
            bs, _, img_h, img_w = x.size()
            y = F.one_hot(y.long(), self.num_classes).float()
            y = y[..., None, None]
            y = y.expand((bs, self.num_classes, img_h, img_w))

            x = self._conv1_1(x)
            y = self._conv1_2(y)

            x = torch.cat([x, y], 1)
        x = self.d(x)
        return x

class Generator3(nn.Module): # imgsize = 64
    def __init__(self,nc=3, nz = 100,ngf = 64,num_classes = 10,condition=False):
        super().__init__()
        self.num_classes = num_classes
        self.condition = condition  # 是否使用条件GAN

        if condition:
            self.deconv1_1 = nn.Linear(nz, ngf * 4 * 4 * 4, bias=False)

            self.deconv1_2 = nn.Linear(num_classes, ngf * 4 * 4 * 4, bias=False)

        self.g = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(nz,ngf * 8*4*4,bias=False) if not condition else nn.Identity(),

            Resize(ngf * 8*4*4,4,4),

            # state size. (ngf*8) x 4 x 4
            nn.Upsample(scale_factor=2),
            ConvBL(ngf * 8, ngf * 4, 3, 1, 1),

            # state size. (ngf*4) x 8 x 8
            nn.Upsample(scale_factor=2),
            ConvBL(ngf * 4, ngf * 2, 3, 1, 1),

            # state size. (ngf*2) x 16 x 16
            nn.Upsample(scale_factor=2),
            ConvBL(ngf * 2, ngf, 3, 1, 1),

            # state size. (ngf) x 32 x 32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Sigmoid(), # 对应图像 norm (0.,1.)
            # nn.Hardtanh(0, 1, True), # [0.,1.]
            # nn.Tanh() # 对应图像norm -1.0~1.0

        )

        # weights_init(self)

    def forward(self, x,y=None):
        x = x.flatten(1)
        if self.condition:
            y = F.one_hot(y.long(), self.num_classes).float()
            x = self.deconv1_1(x)
            y = self.deconv1_2(y)

            x = torch.cat([x, y], 1)
        return self.g(x)

class GAN3(BaseGAN):
    def __init__(self,nc=3,nz=100,dim=64,num_classes=10,condition=False):
        super().__init__(nc,nz,dim,num_classes,condition)

    def generator(self):
        return Generator3(self.nc,self.nz,self.dim,self.num_classes,self.condition)

    def discriminator(self):
        return Discriminator3(self.nc,self.dim,self.num_classes,self.condition)

# ----------参考：BigGAN----------------------------------------------------------
try:
    from .bigGAN.model_resnet import GBlock,SpectralNorm,SelfAttention,ScaledCrossReplicaBatchNorm2d,spectral_norm
except:
    from bigGAN.model_resnet import GBlock, SpectralNorm, SelfAttention, ScaledCrossReplicaBatchNorm2d,spectral_norm

class Discriminator4(nn.Module): # imgsize = 128
    def __init__(self, nc = 3,chn = 64,num_classes=10,condition=False):
        super().__init__()
        self.num_classes = num_classes
        self.condition = condition # 是否使用条件GAN

        def conv(in_channel, out_channel, downsample=True):
            return GBlock(in_channel, out_channel,
                          bn=False,
                          upsample=False, downsample=downsample)

        self.d = nn.Sequential(
            nn.Sequential(SpectralNorm(nn.Conv2d(nc, 1 * chn, 3, padding=1), ),
                          nn.ReLU(),
                          SpectralNorm(nn.Conv2d(1 * chn, 1 * chn, 3, padding=1), ),
                          nn.AvgPool2d(2)),

            SpectralNorm(nn.Conv2d(nc, 1 * chn, 1)),

            nn.Sequential(conv(1 * chn, 1 * chn, downsample=True),
                          SelfAttention(1 * chn),
                          conv(1 * chn, 2 * chn, downsample=True),
                          conv(2 * chn, 4 * chn, downsample=True),
                          conv(4 * chn, 8 * chn, downsample=True),
                          conv(8 * chn, 16 * chn, downsample=True),
                          conv(16 * chn, 16 * chn, downsample=False)),

            SpectralNorm(nn.Linear(16 * chn, 1)),
        )

        self.embed = nn.Embedding(num_classes, 16 * chn)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = spectral_norm(self.embed)

        # weights_init(self)

    def forward(self, x,y=None):
        out = self.d[0](x)
        out = out + self.d[1](F.avg_pool2d(x, 2))
        out = self.d[2](out)
        out = F.relu(out)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.sum(2)

        out_linear = self.d[3](out).squeeze(1)
        embed = self.embed(y.long())

        prod = (out * embed).sum(1)

        return out_linear + prod

class Generator4(nn.Module): # imgsize = 128
    def __init__(self,nc=3, nz = 120,ngf = 64,num_classes = 10,condition=False):
        super().__init__()
        self.num_classes = num_classes
        self.condition = condition  # 是否使用条件GAN

        self.fc = SpectralNorm(nn.Linear(num_classes, 128, bias=False))

        self.g = nn.Sequential(
            SpectralNorm(nn.Linear(20, 4 * 4 * 16 * ngf)),
            Resize(4 * 4 * 16 * ngf, 4, 4),

            GBlock(16*ngf,16*ngf,n_class=num_classes),
            GBlock(16*ngf,8*ngf,n_class=num_classes),
            GBlock(8*ngf,4*ngf,n_class=num_classes),
            GBlock(4*ngf,2*ngf,n_class=num_classes), # 5
            SelfAttention(2 * ngf),
            GBlock(2 * ngf, 1 * ngf, n_class=num_classes), # 7

            ScaledCrossReplicaBatchNorm2d(1 * ngf),
            nn.ReLU(),
            SpectralNorm(nn.Conv2d(1 * ngf, nc, [3, 3], padding=1)),
            nn.Sigmoid(), # 对应图像 norm (0.,1.)
            # nn.Tanh() # 对应图像norm -1.0~1.0
        )

        # weights_init(self)

    def forward(self, x,y=None):
        x = x.flatten(1)
        codes = torch.split(x, 20, 1)
        class_emb = self.fc(F.one_hot(y.long(), self.num_classes).float())

        out = self.g[:2](codes[0])

        condition = torch.cat([codes[1], class_emb], 1)
        out = self.g[2](out, condition)

        condition = torch.cat([codes[2], class_emb], 1)
        out = self.g[3](out, condition)

        condition = torch.cat([codes[3], class_emb], 1)
        out = self.g[4](out, condition)

        condition = torch.cat([codes[4], class_emb], 1)
        out = self.g[5](out, condition)

        out = self.g[6](out)

        condition = torch.cat([codes[5], class_emb], 1)
        out = self.g[7](out, condition)

        out = self.g[8:](out)

        return out

class GAN4(BaseGAN):
    def __init__(self,nc=3,nz=120,dim=64,num_classes=10,condition=True):
        super().__init__(nc,nz,dim,num_classes,condition)
        assert condition==True

    def generator(self):
        return Generator4(self.nc,self.nz,self.dim,self.num_classes,self.condition)

    def discriminator(self):
        return Discriminator4(self.nc,self.dim,self.num_classes,self.condition)


# --------基于GAN4 尺度缩减一倍 128->64 ------------------------------------------------------------------
class Discriminator5(nn.Module): # imgsize = 64
    def __init__(self, nc = 3,chn = 64,num_classes=10,condition=False):
        super().__init__()
        self.num_classes = num_classes
        self.condition = condition # 是否使用条件GAN

        def conv(in_channel, out_channel, downsample=True):
            return GBlock(in_channel, out_channel,
                          bn=False,
                          upsample=False, downsample=downsample)

        self.d = nn.Sequential(
            nn.Sequential(SpectralNorm(nn.Conv2d(nc, 1 * chn, 3, padding=1), ),
                          nn.ReLU(),
                          SpectralNorm(nn.Conv2d(1 * chn, 1 * chn, 3, padding=1), ),
                          nn.AvgPool2d(2)),

            SpectralNorm(nn.Conv2d(nc, 1 * chn, 1)),

            nn.Sequential(conv(1 * chn, 1 * chn, downsample=True),
                          SelfAttention(1 * chn),
                          conv(1 * chn, 2 * chn, downsample=True),
                          conv(2 * chn, 4 * chn, downsample=True),
                          conv(4 * chn, 8 * chn, downsample=True),
                          conv(8 * chn, 16 * chn, downsample=True),
                          # conv(16 * chn, 16 * chn, downsample=False)
                          ),

            SpectralNorm(nn.Linear(16 * chn, 1)),
        )

        self.embed = nn.Embedding(num_classes, 16 * chn)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = spectral_norm(self.embed)

        # weights_init(self)

    def forward(self, x,y=None):
        out = self.d[0](x)
        out = out + self.d[1](F.avg_pool2d(x, 2))
        out = self.d[2](out)
        out = F.relu(out)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.sum(2)

        out_linear = self.d[3](out).squeeze(1)
        embed = self.embed(y.long())

        prod = (out * embed).sum(1)

        return out_linear + prod

class Generator5(nn.Module): # imgsize = 64
    def __init__(self,nc=3, nz = 120,ngf = 64,num_classes = 10,condition=False):
        super().__init__()
        self.num_classes = num_classes
        self.condition = condition  # 是否使用条件GAN

        self.fc = SpectralNorm(nn.Linear(num_classes, 128, bias=False))

        self.g = nn.Sequential(
            SpectralNorm(nn.Linear(20, 4 * 4 * 16 * ngf)),
            Resize(4 * 4 * 16 * ngf, 4, 4),

            # GBlock(16*ngf,16*ngf,n_class=num_classes),
            nn.Identity(),
            GBlock(16*ngf,8*ngf,n_class=num_classes),
            GBlock(8*ngf,4*ngf,n_class=num_classes),
            GBlock(4*ngf,2*ngf,n_class=num_classes), # 5
            SelfAttention(2 * ngf),
            GBlock(2 * ngf, 1 * ngf, n_class=num_classes), # 7

            ScaledCrossReplicaBatchNorm2d(1 * ngf),
            nn.ReLU(),
            SpectralNorm(nn.Conv2d(1 * ngf, nc, [3, 3], padding=1)),
            nn.Sigmoid(), # 对应图像 norm (0.,1.)
            # nn.Tanh() # 对应图像norm -1.0~1.0
        )

        # weights_init(self)

    def forward(self, x,y=None):
        x = x.flatten(1)
        codes = torch.split(x, 20, 1)
        class_emb = self.fc(F.one_hot(y.long(), self.num_classes).float())

        out = self.g[:2](codes[0])

        # condition = torch.cat([codes[1], class_emb], 1)
        # out = self.g[2](out, condition)

        condition = torch.cat([codes[2], class_emb], 1)
        out = self.g[3](out, condition)

        condition = torch.cat([codes[3], class_emb], 1)
        out = self.g[4](out, condition)

        condition = torch.cat([codes[4], class_emb], 1)
        out = self.g[5](out, condition)

        out = self.g[6](out)

        condition = torch.cat([codes[5], class_emb], 1)
        out = self.g[7](out, condition)

        out = self.g[8:](out)

        return out

class GAN5(BaseGAN):
    def __init__(self,nc=3,nz=120,dim=64,num_classes=10,condition=True):
        super().__init__(nc,nz,dim,num_classes,condition)
        assert condition==True

    def generator(self):
        return Generator5(self.nc,self.nz,self.dim,self.num_classes,self.condition)

    def discriminator(self):
        return Discriminator5(self.nc,self.dim,self.num_classes,self.condition)



if __name__=="__main__":
    x = torch.rand([32,1,64,64])
    # x = torch.rand([32,120,1,1])
    labels = torch.randint(0, 10, [32], dtype=torch.long)

    m = Discriminator5(nc=1)
    # m = Generator5(nc=1)
    print(m(x,labels).shape)