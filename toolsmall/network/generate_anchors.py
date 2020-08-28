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


if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchorsV2([128,256,512],[0.5,1,2])
    # a = generate_anchors()
    # print(time.time() - t)
    print(a)
    exit(0)
    # from IPython import embed; embed()
    anchors,keep = getAnchorsV2([100,60],10)
    # anchors,keep = getAnchors([38,50],16) # ratios=(1,)
    print(time.time() - t)
    print(len(anchors))
    print(anchors[-20:])