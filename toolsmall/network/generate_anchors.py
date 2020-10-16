from __future__ import print_function
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
import pdb
from itertools import product
import torch,math
from math import ceil

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# torchvision.model.detection.rpn.AnchorGenerator
# TODO: https://github.com/pytorch/pytorch/issues/26792
# For every (aspect_ratios, scales) combination, output a zero-centered anchor with those values.
# (scales, aspect_ratios) are usually an element of zip(self.scales, self.aspect_ratios)
# import torch
# sizes=(128, 256, 512),
# aspect_ratios=(0.5, 1.0, 2.0),
def generate_anchorsV2(scales, aspect_ratios, dtype=np.float32):
    # type: (List[int], List[float], int, Device)  # noqa: F821
    scales = np.array(scales, dtype=dtype)
    aspect_ratios = np.array(aspect_ratios, dtype=dtype)
    h_ratios = np.sqrt(aspect_ratios)
    w_ratios = 1.0 / h_ratios

    ws = (w_ratios[:, None] * scales[None, :]).reshape(-1)
    hs = (h_ratios[:, None] * scales[None, :]).reshape(-1)

    base_anchors = np.stack([-ws, -hs, ws, hs], axis=1) / 2
    return base_anchors.round()

"""
与上面的 generate_anchorsV2 几乎等价
"""
def generate_cell_anchors(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
    """
    Generate a tensor storing canonical anchor boxes, which are all anchor
    boxes of different sizes and aspect_ratios centered at (0, 0).
    We can later build the set of anchors for a full feature map by
    shifting and tiling these tensors (see `meth:_grid_anchors`).

    Args:
        sizes (tuple[float]):
        aspect_ratios (tuple[float]]):

    Returns:
        Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
            in XYXY format.
    """

    # This is different from the anchor generator defined in the original Faster R-CNN
    # code or Detectron. They yield the same AP, however the old version defines cell
    # anchors in a less natural way with a shift relative to the feature grid and
    # quantization that results in slightly different sizes for different aspect ratios.
    # See also https://github.com/facebookresearch/Detectron/issues/227

    anchors = []
    for size in sizes:
        area = size ** 2.0
        for aspect_ratio in aspect_ratios:
            # s * s = w * h
            # a = h / w
            # ... some algebra ...
            # w = sqrt(s * s / a)
            # h = a * w
            w = math.sqrt(area / aspect_ratio)
            h = aspect_ratio * w
            x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
            anchors.append([x0, y0, x1, y1])
    return torch.tensor(anchors)

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    return anchors # [x1,y1,x2,y2]

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


"""
使用论文给出的方式生成anchors
输入大小是把图片最小边resize到 s=600
"""
def anchorsWH(baseSize=600,scales=[128,256,512],ratios=[0.5,1,2]):
    # 如果最小边resize到800则按比例缩放
    scales = [i*baseSize/600 for i in scales]

    # h,w
    anchors_wh = []
    for s in scales:
        for r in ratios:
               anchors_wh.append([s,s*r])

    return np.array(anchors_wh,np.float32)

def getAnchors(fmap=[14,14],stride=16,baseSize=600,scales=[128,256,512],ratios=[0.5,1,2]):
    """对应到 convolutional feature map
        并删除掉越界的 anchors
    """

    anchors_wh = anchorsWH(baseSize,scales,ratios)/stride # 缩放到feature map
    """
    anchor =[]
    for i,j in product(range(fmap[0]),range(fmap[1])):
        for (w,h) in anchors_wh:
            # j,i,w,h -> cx,cy,w,h
            x1, y1, x2, y2 = j-w/2,i-h/2,j+w/2,i+h/2
            # 剔除掉越界的
            if x1<0 or y1<0 or x2 > fmap[1] or y2 >fmap[0]:continue
            anchor.append([x1, y1, x2, y2])
    anchor = np.asarray(anchor)
    """
    X,Y=np.meshgrid(np.arange(0,fmap[1]),np.arange(0,fmap[0]))
    x1y1x2y2=np.concatenate((X[..., None], Y[..., None],X[..., None], Y[..., None]), -1).astype(np.float32)
    anchors_whwh = np.concatenate((-1*anchors_wh/2,anchors_wh/2),1)
    anchor = (x1y1x2y2[:,:,None,:]+anchors_whwh[None,None,...]).reshape(-1,4)
    # 剔除掉越界的
    keep = np.stack((anchor[:,0]>0,anchor[:,1]>0,anchor[:,2]<fmap[1],anchor[:,3]<fmap[0]),-1)
    keep = np.sum(keep,1)==4
    anchor = anchor[keep]
    # """
    return anchor,keep

def getAnchorsV2(fmap=[14,14],stride=16,baseSize=600,scales=[128,256,512],ratios=[0.5,1,2]):
    """对应到 convolutional feature map
        并删除掉越界的 anchors
    """
    # base_size = 16
    # x1y1x2y2_00 = generate_anchors(base_size*baseSize/600,ratios,np.array(scales,np.float32)/base_size)/stride
    x1y1x2y2_00 = generate_anchorsV2([i*baseSize/600 for i in scales],ratios)/stride

    X,Y=np.meshgrid(np.arange(0,fmap[1]),np.arange(0,fmap[0]))
    x1y1x2y2=np.concatenate((X[..., None], Y[..., None],X[..., None], Y[..., None]), -1).astype(np.float32)
    anchor = (x1y1x2y2[:,:,None,:]+x1y1x2y2_00[None,None,...]).reshape(-1,4)
    # 剔除掉越界的
    keep = np.stack((anchor[:,0]>0,anchor[:,1]>0,anchor[:,2]<fmap[1],anchor[:,3]<fmap[0]),-1)
    keep = np.sum(keep,1)==4
    anchor = anchor[keep]
    # """
    return anchor,keep


def getAnchorsV2_s(resize=[320,320],stride=16,scales=[128,256,512],ratios=[0.5,1,2]):
    """缩放到到 0~1
        不删除掉越界的 anchors
    """
    baseSize = resize[0]
    fmap = [re//stride for re in resize]
    # base_size = 16
    # x1y1x2y2_00 = generate_anchors(base_size*baseSize/600,ratios,np.array(scales,np.float32)/base_size)/stride
    x1y1x2y2_00 = generate_anchorsV2([i*baseSize/600 for i in scales],ratios)/stride

    X,Y=np.meshgrid(np.arange(0,fmap[1]),np.arange(0,fmap[0]))
    x1y1x2y2=np.concatenate((X[..., None], Y[..., None],X[..., None], Y[..., None]), -1).astype(np.float32)
    anchor = (x1y1x2y2[:,:,None,:]+x1y1x2y2_00[None,None,...]).reshape(-1,4)
    # 缩减到 0~1
    anchor /= np.array((fmap[1],fmap[0],fmap[1],fmap[0]),dtype=np.float32)[None,...]

    # clip
    anchor = np.clip(anchor,0,1)

    # 剔除掉越界的
    # keep = np.stack((anchor[:,0]>0,anchor[:,1]>0,anchor[:,2]<fmap[1],anchor[:,3]<fmap[0]),-1)
    # keep = np.sum(keep,1)==4
    # anchor = anchor[keep]
    # """
    return anchor#,keep


def getAnchorsV3(fmap=[14,14],stride=16,anchors_wh:np.array=None):
    """对应到 convolutional feature map
        并删除掉越界的 anchors
    """
    anchors_wh = anchors_wh/stride # 缩放到feature map
    """
    anchor =[]
    for i,j in product(range(fmap[0]),range(fmap[1])):
        for (w,h) in anchors_wh:
            # j,i,w,h -> cx,cy,w,h
            x1, y1, x2, y2 = j-w/2,i-h/2,j+w/2,i+h/2
            # 剔除掉越界的
            if x1<0 or y1<0 or x2 > fmap[1] or y2 >fmap[0]:continue
            anchor.append([x1, y1, x2, y2])
    anchor = np.asarray(anchor)
    """
    X,Y=np.meshgrid(np.arange(0,fmap[1]),np.arange(0,fmap[0]))
    x1y1x2y2=np.concatenate((X[..., None], Y[..., None],X[..., None], Y[..., None]), -1).astype(np.float32)
    anchors_whwh = np.concatenate((-1*anchors_wh/2,anchors_wh/2),1)
    anchor = (x1y1x2y2[:,:,None,:]+anchors_whwh[None,None,...]).reshape(-1,4)
    # 剔除掉越界的
    keep = np.stack((anchor[:,0]>0,anchor[:,1]>0,anchor[:,2]<fmap[1],anchor[:,3]<fmap[0]),-1)
    keep = np.sum(keep,1)==4
    anchor = anchor[keep]
    # """
    return anchor,keep

# -------------------------------------------------------------------------------------------------
# 对应ssd的先验anchor(6个默认anchor)
def getAnchorsWH(featureMap_idx=1,num_featureMap=4,scales_min=0.2,scales_max=0.9,ratios=[1,2,3,1/2,1/3]):
    """
    :param featureMap_idx: 特征map的索引 1~num_featureMap
    :param num_featureMap: 特征map的个数
    :param scales_min:
    :param scales_max:
    :param ratios:
    :return: 缩减到 0.~1.
    """
    sk = scales_min+(scales_max-scales_min)*(featureMap_idx-1)/(num_featureMap-1)
    sk_1 = scales_min+(scales_max-scales_min)*(featureMap_idx)/(num_featureMap-1)
    wh=[]
    for r in ratios:
        w = sk*np.sqrt(r)
        h = sk/np.sqrt(r)
        wh.append([w,h])
        if r==1:
            sk_2 = np.sqrt(sk*sk_1)
            w = sk_2 * np.sqrt(r)
            h = sk_2 / np.sqrt(r)
            wh.append([w, h])
    return np.asarray(wh).clip(0,1)

"""
x1y1x2y2格式 0~1 效果差(不推荐)
每个锚点对应6个anchor
"""
def getAnchorSSD(resize=(320,320),strides=[4,8,16,32],device="cpu",clip=True,scales_min=0.2,scales_max=0.9,ratios=[1,2,3,1/2,1/3]):
    """:return 缩放到0~1"""
    anchor_list = []
    _strides = [4,8,16,32,64]
    for idx,stride in enumerate(strides):
        idx = _strides.index(stride)+1
        anchors_wh = getAnchorsWH(idx,len(_strides),scales_min,scales_max,ratios) # 0~1
        fh,fw = ceil(resize[0]/stride),ceil(resize[1]/stride)

        X, Y = np.meshgrid(np.arange(0, fw), np.arange(0, fh))
        x1y1x2y2 = np.concatenate((X[..., None], Y[..., None], X[..., None], Y[..., None]), -1).astype(np.float32) + 0.5
        # to 0~1
        x1y1x2y2[...,[0,2]] /= fw
        x1y1x2y2[...,[1,3]] /= fh

        anchors_whwh = np.concatenate((-1 * anchors_wh / 2, anchors_wh / 2), 1)
        anchor = (x1y1x2y2[:, :, None, :] + anchors_whwh[None, None, ...]).reshape(-1, 4) #(x1y1x2y2)

        anchor_list.append(anchor)

    priors = torch.from_numpy(np.concatenate(anchor_list,0)).float().to(device)
    if clip:
        priors.clamp_(max=1, min=0)
        return priors
    else:
        keep = (priors >= 0) * (priors < 1)
        keep = keep.sum(-1) == 4
        return priors, keep


"""
xywh格式,缩减到0~1(缩减到输入图像)
每个锚点对应6个anchor
"""
def get_prior_box(resize=(320,320),strides=[4,8,16,32],device="cpu",clip=True,_priorBox=None):
    if _priorBox is None:
        _priorBox = {
            "min_dim": 300.0,
            "min_sizes": [30, 60, 111, 162],
            "max_sizes": [60, 111, 162, 213],
            "aspect_ratios": [[2, 3], [2, 3], [2, 3], [2, 3]],  # [[2],[2],[2],[2]]
            # "variance":[0.1,0.2],
            # "clip":True,
            # "thred_iou":0.5,
            "strides": [4, 8, 16, 32]
        }

    scales = resize[0]/_priorBox["min_dim"]
    aspect_ratios = _priorBox["aspect_ratios"]
    min_sizes = [size*scales for size in _priorBox["min_sizes"]]
    max_sizes = [size*scales for size in _priorBox["max_sizes"]]
    priors = []
    h, w = resize
    for idx, stride in enumerate(strides):
        idx = _priorBox["strides"].index(stride)
        fh, fw = ceil(h / stride), ceil(w / stride)
        for i in range(fh):
            for j in range(fw):
                # unit center x,y
                cx = (j + 0.5) / fw
                cy = (i + 0.5) / fh

                # small sized square box
                size_min = min_sizes[idx]
                size_max = max_sizes[idx]
                size = size_min
                bh, bw = size / h, size / w
                priors.append([cx, cy, bw, bh])

                # big sized square box
                size = np.sqrt(size_min * size_max)
                bh, bw = size / h, size / w
                priors.append([cx, cy, bw, bh])

                # change h/w ratio of the small sized box
                size = size_min
                bh, bw = size / h, size / w
                for ratio in aspect_ratios[idx]:
                    ratio = np.sqrt(ratio)
                    priors.append([cx, cy, bw * ratio, bh / ratio])
                    priors.append([cx, cy, bw / ratio, bh * ratio])

    priors = torch.tensor(priors, device=device, dtype=torch.float32)
    if clip:
        priors.clamp_(max=1, min=0)
        return priors
    else:
        keep = (priors>=0)*(priors<1)
        keep = keep.sum(-1)==4
        return priors,keep


"""参考 getAnchorsV2

x1y1x2y2 格式,缩减到0~1(缩减到输入图像)
每个锚点对应3个anchor

缩放的大小不一点是固定的
如:
按最小边 缩放到 600,另一个大小限制在 1200
按最小边 缩放到 800,另一个大小限制在 1330
"""
def getAnchorsV2_FPN(resize=(600,600),strides=[4,8,16,32,64,128],scales=[16,32,64,128,256,512],ratios=[0.5,1,2],clip=True,device="cpu"):
    """对应到 convolutional feature map
        并删除掉越界的 anchors
    """
    anchor_list = []
    scales = [i*min(resize)/600 for i in scales]
    for idx,stride in enumerate(strides):
        # 统一缩放到0~1
        x1y1x2y2_00 = generate_anchorsV2([scales[idx]], ratios) / resize[0]
        fh, fw = ceil(resize[0] / stride), ceil(resize[1] / stride)

        X,Y=np.meshgrid(np.arange(0,fw),np.arange(0,fh))
        # 统一缩放到0~1
        x1y1x2y2=np.concatenate((X[..., None], Y[..., None],X[..., None], Y[..., None]), -1)/np.array([[fw,fh,fw,fh]])
        anchor = (x1y1x2y2[:,:,None,:]+x1y1x2y2_00[None,None,...]).reshape(-1,4)

        anchor_list.append(anchor)

    priors = torch.from_numpy(np.concatenate(anchor_list, 0)).float().to(device)

    if clip:
        priors.clamp_(max=1, min=0)
        return priors
    else:
        keep = (priors >= 0) * (priors < 1)
        keep = keep.sum(-1) == 4
        return priors, keep

"""参考 getAnchorsV2_FPN

x1y1x2y2 格式,缩减到0~1(缩减到输入图像)
每个锚点对应9个anchor

缩放的大小不一点是固定的
如:
按最小边 缩放到 600,另一个大小限制在 1200
按最小边 缩放到 800,另一个大小限制在 1330
"""
def getAnchorsV2_FPN_V2(resize=(600,600),strides=[4,8,16,32,64,128],scales=[16,32,64,128,256,512],ratios=[0.5,1,2],clip=True,device="cpu"):
    """对应到 convolutional feature map
        并删除掉越界的 anchors
    """
    anchor_list = []
    scales = [i*min(resize)/600 for i in scales]
    for idx,stride in enumerate(strides):
        # 统一缩放到0~1
        x1y1x2y2_00 = generate_anchorsV2([scales[idx],scales[idx]*2**(1/3),scales[idx]*2**(2/3)], ratios) / resize[0]
        fh, fw = ceil(resize[0] / stride), ceil(resize[1] / stride)

        X,Y=np.meshgrid(np.arange(0,fw),np.arange(0,fh))
        # 统一缩放到0~1
        x1y1x2y2=np.concatenate((X[..., None], Y[..., None],X[..., None], Y[..., None]), -1)/np.array([[fw,fh,fw,fh]])
        anchor = (x1y1x2y2[:,:,None,:]+x1y1x2y2_00[None,None,...]).reshape(-1,4)

        anchor_list.append(anchor)

    priors = torch.from_numpy(np.concatenate(anchor_list, 0)).float().to(device)

    if clip:
        priors.clamp_(max=1, min=0)
        return priors
    else:
        keep = (priors >= 0) * (priors < 1)
        keep = keep.sum(-1) == 4
        return priors, keep

if __name__ == '__main__':
    # print(generate_cell_anchors((32,64),(0.5,1,2)))
    # print("----------------")
    # print(generate_anchorsV2((32,64),(0.5,1,2)))

    anchors_x1y1x2y2 = getAnchorsV2_FPN((300,300),[8,16,32],(64,128,256))
    print()
    # anchors_x1y1x2y2=getAnchorSSD(clip=True)
    # print()
    # import time
    # t = time.time()
    # a = generate_anchorsV2([128,256,512],[0.5,1,2])
    # # a = generate_anchors()
    # # print(time.time() - t)
    # print(a)
    # exit(0)
    # # from IPython import embed; embed()
    # anchors,keep = getAnchorsV2([100,60],10)
    # # anchors,keep = getAnchors([38,50],16) # ratios=(1,)
    # print(time.time() - t)
    # print(len(anchors))
    # print(anchors[-20:])