# import torch
# from torch import Tensor
# from torch.jit.annotations import List, Optional, Dict, Tuple
import numpy as np
import torch,math
from math import ceil

# 最原始的anchor生成方式
def generate_anchors(anchor_size = (32,64,128,256,512),aspect_ratios=(0.5,1.0,2.0)):
    # 返回 以（0,0）为中心的 先验anchor
    # nums = len(anchor_size)*len(aspect_ratios) # 每个锚点产生多少个anchor
    wh = []
    for size in anchor_size:
        for ratios in aspect_ratios:
            wh.append((size,size*ratios))
    wh = np.array(wh)

    x1y1x2y2 = np.concatenate((-wh,wh),1)/2.0 # 以（0,0）为中心的 先验anchor

    return x1y1x2y2

# [x1,y1,x2,y2] ,输入图像大小
def generate_anchorsV2(scales, aspect_ratios, dtype=np.float32):
    scales = np.array(scales, dtype=dtype)
    aspect_ratios = np.array(aspect_ratios, dtype=dtype)
    h_ratios = np.sqrt(aspect_ratios)
    w_ratios = 1.0 / h_ratios

    ws = (w_ratios[:, None] * scales[None, :]).reshape(-1)
    hs = (h_ratios[:, None] * scales[None, :]).reshape(-1)

    base_anchors = np.stack([-ws, -hs, ws, hs], axis=1) / 2
    return base_anchors.round()

# 给定预设值anchor的大小
# [x1,y1,x2,y2] ,输入图像大小
def generate_anchorsV3(input_size:list,wh:np.array): # 给定预设值anchor的大小
    # wh 缩放到0~1
    h,w = input_size # 输入的大小
    wh = wh * np.array([w,h])[None]
    ws = wh[:,0]
    hs = wh[:,1]
    base_anchors = np.stack([-ws, -hs, ws, hs], axis=1) / 2
    return base_anchors.round()

# [x1,y1,x2,y2]格式,0~1
def getAnchors(heatMap_shape=(54,70),stride=16,anchor_size = (32,64,128,256,512),
               aspect_ratios=(0.5,1.0,2.0),use_cell_center=False,fixsize=False):
    if fixsize: # ssd yolo
        resize_size = heatMap_shape[0]*stride
        base_anchors = generate_anchorsV2([resize_size*size/600 for size in anchor_size],aspect_ratios)
    else: # faster rcnn /rfcn
        base_anchors = generate_anchorsV2(anchor_size,aspect_ratios) # # 以（0,0）为中心的 先验anchor (x1,y1,x2,y2)
    h,w = heatMap_shape
    X, Y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    x1y1x2y2 = np.concatenate((X[..., None], Y[..., None], X[..., None], Y[..., None]), -1).astype(np.float32)*stride # 对应到输入图像上
    if use_cell_center: # 使用每个cell的中心，默认是使用左上角
        x1y1x2y2 += stride/2.0

    anchor = (x1y1x2y2[:, :, None, :] + base_anchors[None, None, ...]).reshape(-1, 4) # [h,w,a,4] -->[-1,4]

    # normalization
    anchor = anchor / (np.array([w, h, w, h])[None] * stride)
    # 裁剪到图像内
    anchor = np.clip(anchor,0,1)

    return anchor

# 使用多层RPN
# [x1,y1,x2,y2]格式,0~1
def getAnchors_FPN(heatMap_shape=((112,144),(56,72),(28,36),(14,18),(7,9)),stride=(4,8,16,32,64),
                   anchor_size=((32,),(64,),(128,),(256,),(512,)),aspect_ratios=(0.5,1.0,2.0),
                   use_cell_center=False,fixsize=False):
    nums_ht = len(anchor_size)
    anchor_fpn = []
    for i in range(nums_ht):
        anchor_fpn.append(getAnchors(heatMap_shape[i],stride[i],anchor_size[i],
                                     aspect_ratios,use_cell_center,fixsize))

    return np.concatenate(anchor_fpn,0)


# ---------ssd anchor------------------------------------------------------
# [x1,y1,x2,y2]格式 缩放到0~1
def getAnchorsV2_s(resize=[320,320],stride=16,scales=[128,256,512],ratios=[0.5,1,2]):
    """缩放到到 0~1
        不删除掉越界的 anchors
    """
    baseSize = resize[0]
    fmap = [re//stride for re in resize]
    x1y1x2y2_00 = generate_anchorsV2([i*baseSize/600 for i in scales],ratios)/stride

    X,Y=np.meshgrid(np.arange(0,fmap[1]),np.arange(0,fmap[0]))
    x1y1x2y2=np.concatenate((X[..., None], Y[..., None],X[..., None], Y[..., None]), -1).astype(np.float32)
    anchor = (x1y1x2y2[:,:,None,:]+x1y1x2y2_00[None,None,...]).reshape(-1,4)
    # 缩减到 0~1
    anchor /= np.array((fmap[1],fmap[0],fmap[1],fmap[0]),dtype=np.float32)[None,...]

    # clip
    anchor = np.clip(anchor,0,1)

    return anchor


# [x,y,w,h]格式 缩放到0~1
def get_prior_box(resize=(320,320),strides=[4,8,16,32],clip=True,_priorBox=None):
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

    # priors = torch.tensor(priors, device=device, dtype=torch.float32)

    if clip:
        # priors.clamp_(max=1, min=0)
        priors = np.clip(priors,0,1)

        return priors


if __name__=="__main__":
    # print(generate_anchors((32,64,128,256,512),(0.5,1.0,2.0)))
    # print(generate_anchorsV2((32,64,128,256,512),(0.5,1.0,2.0)))

    # getAnchors(use_cell_center=True,use_clip=True)
    getAnchors_FPN()