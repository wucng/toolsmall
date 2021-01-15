import os
import torch
import numpy as np
import random
import cv2,math
import PIL.Image
from torchvision.ops.boxes import batched_nms
from torch.nn import functional as F
from itertools import product

def glob_format(path,fmt_list = ('.jpg', '.jpeg', '.png',".xml"),base_name = False):
    #print('--------pid:%d start--------------' % (os.getpid()))
    fs = []
    if not os.path.exists(path):return fs
    for root, dirs, files in os.walk(path):
        for file in files:
            item = os.path.join(root, file)
            # item = unicode(item, encoding='utf8')
            fmt = os.path.splitext(item)[-1]
            if fmt.lower() not in fmt_list:
                # os.remove(item)
                continue
            if base_name:fs.append(file)  # fs.append(os.path.splitext(file)[0])
            else:fs.append(item)
    #print('--------pid:%d end--------------' % (os.getpid()))
    return fs


def batch(imgs:list,stride=32):
    nums = len(imgs)
    new_imgs = imgs
    # for i in range(nums):
    #     new_imgs.append(resizeMinMax(imgs[i],min_size,max_size))

    # 获取最大的 高和宽
    max_h = max([img.size(1) for img in new_imgs])
    max_w = max([img.size(2) for img in new_imgs])

    # 扩展到 stride的倍数(32倍数 GPU可以加速，且网络的最终stride=32)
    max_h = int(np.ceil(1.0*max_h /stride) *stride)
    max_w = int(np.ceil(1.0*max_w /stride) *stride)

    # 初始一个tensor 用于填充
    batch_img = torch.ones([nums,3,max_h,max_w],device=imgs[0].device)*114
    for i,img in enumerate(new_imgs):
        c,h,w = img.size()
        batch_img[i,:,:h,:w] = img  # 从左上角往下填充

    return batch_img

def collate_fn(batch_data):
    data_list = []
    target_list = []
    for data,target in batch_data:
        data_list.append(data)
        target_list.append(target)

    return data_list,target_list

def xywh2x1y1x2y2(boxes):
    """
    xywh->x1y1x2y2

    :param boxes: [...,4]
    :return:
    """
    x1y1=boxes[...,:2]-boxes[...,2:]/2
    x2y2=boxes[...,:2]+boxes[...,2:]/2

    return torch.cat((x1y1,x2y2),-1)

def x1y1x2y22xywh(boxes):
    """
    x1y1x2y2-->xywh

    :param boxes: [...,4]
    :return:
    """
    xy=(boxes[...,:2]+boxes[...,2:])/2
    wh=boxes[...,2:]-boxes[...,:2]

    return torch.cat((xy,wh),-1)


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x0, y0, x1, y1) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x0, y0, x1, y1) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


# fastercnn/rfcn
def propose_boxes(logits, bbox_reg, anchors, training,targets,device,
                  rpn_pre_nms_top_n_train = 12000,rpn_pre_nms_top_n_test=6000,
                  rpn_post_nms_top_n_train = 2000,rpn_post_nms_top_n_test = 1000,
                  rpn_nms_thresh=0.7
                  ):
    bs = logits.size(0)
    input_h, input_w, scale = targets[0]["resize"]

    # Normalize 0~1
    anchors_x1y1x2y2 = anchors

    # 选择scores最大的前12000个候选区域
    if training:
        index = logits.sort(1, True)[1][:, :rpn_pre_nms_top_n_train]
    else:
        index = logits.sort(1, True)[1][:, :rpn_pre_nms_top_n_test]

    # candidate_box_x1y1x2y2 = candidate_box_x1y1x2y2[index]
    anchors_x1y1x2y2 = torch.stack([anchors_x1y1x2y2[i][index[i]] for i in range(bs)], 0)
    logits = torch.stack([logits[i][index[i]] for i in range(bs)], 0)
    bbox_reg = torch.stack([bbox_reg[i][index[i]] for i in range(bs)], 0)

    # nms 过滤
    # keep = batched_nms(candidate_box_x1y1x2y2,logits_score,torch.ones_like(logits_score),rpn_nms_thresh)
    if training:
        keep = torch.stack([batched_nms(anchors_x1y1x2y2[i], logits[i],
                                        torch.ones_like(logits[i]), rpn_nms_thresh)[:rpn_post_nms_top_n_train]
                            for i in range(bs)], 0)
    else:
        keep = torch.stack([batched_nms(anchors_x1y1x2y2[i], logits[i],
                                        torch.ones_like(logits[i]), rpn_nms_thresh)[:rpn_post_nms_top_n_test]
                            for i in range(bs)], 0)

    anchors_x1y1x2y2 = torch.stack([anchors_x1y1x2y2[i][keep[i]] for i in range(bs)], 0)

    logits = torch.stack([logits[i][keep[i]] for i in range(bs)], 0)
    bbox_reg = torch.stack([bbox_reg[i][keep[i]] for i in range(bs)], 0)

    # anchors修正得到候选框
    anchors_xywh = x1y1x2y22xywh(anchors_x1y1x2y2)
    candidate_box_xywh = torch.zeros_like(anchors_xywh)
    candidate_box_xywh[..., :2] = (anchors_xywh[..., 2:] * bbox_reg[..., :2]) + anchors_xywh[..., :2]
    candidate_box_xywh[..., 2:] = anchors_xywh[..., 2:] * torch.exp(bbox_reg[..., 2:])
    proposal = xywh2x1y1x2y2(candidate_box_xywh).clamp(0., 1.)  # 裁剪到图像内

    return anchors_xywh[0], logits[0], bbox_reg[0], proposal[0].detach() * torch.tensor([input_w, input_h, input_w, input_h],
                                                                            dtype=torch.float32, device=device)[None]

def positiveAndNegative(ious, miniBactch=256, rpn=True, logits=None,
                        rpn_fg_iou_thresh=0.7,
                        rpn_bg_iou_thresh = 0.3,
                        rpn_positive_fraction = 0.5,
                        box_fg_iou_thresh=0.5,
                        box_bg_iou_thresh = 0.5,
                        box_positive_fraction = 0.25
):
    # 每个anchor与gt对应的iou
    per_anchor_to_gt, per_anchor_to_gt_index = ious.max(1)
    # 与gt对应的最大IOU的anchor
    per_gt_to_anchor, per_gt_to_anchor_index = ious.max(0)

    # 每个anchor对应的gt
    gt_indexs = per_anchor_to_gt_index

    indexs = torch.ones_like(per_anchor_to_gt) * (-1)

    if rpn:
        indexs[per_anchor_to_gt > rpn_fg_iou_thresh] = 1  # 正样本
        indexs[per_anchor_to_gt < rpn_bg_iou_thresh] = 0  # 负样本

        # 随机选择256个anchors,正负样本比例为1:1
        new_positive = int(miniBactch * rpn_positive_fraction)

    else:  # rcnn
        indexs[per_anchor_to_gt >= box_fg_iou_thresh] = 1  # 正样本
        indexs[torch.bitwise_and(per_anchor_to_gt < box_bg_iou_thresh, per_anchor_to_gt > 0.1)] = 0  # 负样本

        #  25% 的前景,75% 的背景
        new_positive = int(miniBactch * box_positive_fraction)

    # 与gt对应的最大IOU的anchor 也是正样本
    for i, idx in enumerate(per_gt_to_anchor_index):
        indexs[idx] = 1
        gt_indexs[idx] = i

    # -1 忽略
    # nums_positive = (indexs == 1).sum()
    # nums_negative = (indexs == 0).sum()
    idx_positive = torch.nonzero(indexs == 1).squeeze(-1)
    idx_negative = torch.nonzero(indexs == 0).squeeze(-1)
    nums_positive = len(idx_positive)
    nums_negative = len(idx_negative)

    new_negative = miniBactch - min(nums_positive, new_positive)
    # new_negative = 3*min(nums_positive,new_positive) # 正负样本固定为1:3

    if logits is None:  # 随机选
        if nums_positive < new_positive:
            # 选择负样本个数为
            negindex = list(range(nums_negative))
            random.shuffle(negindex)
            indexs[idx_negative[negindex[new_negative:]]] = -1

        elif nums_positive > new_positive:
            posindex = list(range(nums_positive))
            random.shuffle(posindex)
            indexs[idx_positive[posindex[new_positive:]]] = -1

            negindex = list(range(nums_negative))
            random.shuffle(negindex)
            indexs[idx_negative[negindex[new_negative:]]] = -1
        else:
            negindex = list(range(nums_negative))
            random.shuffle(negindex)
            indexs[idx_negative[negindex[new_negative:]]] = -1

    else:  # 根据负样本的loss 选 从大到小选
        if nums_positive < new_positive:
            with torch.no_grad():
                # loss = -F.log_softmax(confidence, dim=1)[:, 0]
                loss = -F.logsigmoid(logits[idx_negative])
            # 从大到小排序
            negindex = loss.sort(descending=True)[1]
            indexs[idx_negative[negindex[new_negative:]]] = -1

        elif nums_positive > new_positive:
            posindex = list(range(nums_positive))
            random.shuffle(posindex)
            indexs[idx_positive[posindex[new_positive:]]] = -1

            with torch.no_grad():
                # loss = -F.log_softmax(confidence, dim=1)[:, 0]
                loss = -F.logsigmoid(logits[idx_negative])
            # 从大到小排序
            negindex = loss.sort(descending=True)[1]
            indexs[idx_negative[negindex[new_negative:]]] = -1
        else:
            with torch.no_grad():
                # loss = -F.log_softmax(confidence, dim=1)[:, 0]
                loss = -F.logsigmoid(logits[idx_negative])
            # 从大到小排序
            negindex = loss.sort(descending=True)[1]
            indexs[idx_negative[negindex[new_negative:]]] = -1

    return indexs, gt_indexs



def _nms(heat, kernel=3):
    """
    :param heat: torch.tensor [bs,c,h,w]
    :param kernel:
    :return:
    """
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


# -----------------------centernet----------------------------------------------
def heatmap2index(heatmap:torch.tensor,heatmap2:torch.tensor=None,thres=0.5,has_background=False):
    """
    heatmap[0,0,:]只能属于一个类别
    :param heatmap: [bs,h,w,num_classes] or [h,w,num_classes]
    :param heatmap2 : [bs,h,w,num_classes,4] or [h,w,num_classes,4]
    :param threds:
    :param has_background: 是否包含背景
    :return: scores, labels,cycx(中心点坐标)
    """
    scores,labels = heatmap.max(-1) # [bs,h,w] or [h,w]
    if has_background:
        keep = torch.bitwise_and(scores > thres, labels > 0)  # 0为背景，包括背景
    else:
        keep = scores > thres
    scores, labels = scores[keep], labels[keep]
    cycx = torch.nonzero(keep)
    if heatmap2 is not None:
        heatmap2 = heatmap2[keep,labels]

    return scores, labels,cycx,keep,heatmap2

def heatmap2indexV2(heatmap:torch.tensor,heatmap2:torch.tensor=None,thres=0.5,has_background=False,topK=5):
    """
    heatmap[0,0,:]可以属于多个类别
    :param heatmap: [bs,h,w,num_classes] or [h,w,num_classes]
    :param heatmap2 : [bs,h,w,num_classes,4] or [h,w,num_classes,4]
    """
    scores, labels = heatmap.topk(topK,-1)

    if heatmap2 is not None:
        h,w,c = labels.shape
        new_heatmap2 = torch.zeros((h,w,c,heatmap2.shape[-1]),device=heatmap2.device)
        # for i in range(h):
        #     for j in range(w):
        #         for k in range(c):
        #             l = labels[i,j,k]
        #             new_heatmap2[i,j,k,:] = heatmap2[i,j,l,:]
        # for i in range(h):
        #     for j in range(w):
        for i,j in product(range(h),range(w)):
                new_heatmap2[i,j] = heatmap2[i,j,labels[i,j]]

    if has_background:
        keep = torch.bitwise_and(scores > thres, labels > 0)  # 0为背景，包括背景
    else:
        keep = scores > thres
    scores, labels = scores[keep], labels[keep]
    cycx = torch.nonzero(keep)[...,:2]
    if heatmap2 is not None:
        new_heatmap2 = new_heatmap2[keep]
        heatmap2= new_heatmap2

    return scores, labels,cycx,keep,heatmap2


def gaussian_radius(det_size, min_overlap=0.7):
    """
    :param det_size: boxes的[h,w]，已经所放到heatmap上
    :param min_overlap:
    :return:
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    :param heatmap: [128,128]
    :param center: [x,y]
    :param radius: int
    :param k:
    :return:
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                      radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom,
                 radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap


def draw_msra_gaussian(heatmap, center, sigma):
    """
    :param heatmap: [128,128]
    :param center: [x,y]
    :param sigma: int
    :return:
    """
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap

# ----------------------------------------------------


"""centernet 的方式标记 heatmap (只标记中心点及其附近点)"""
def drawHeatMapV1(hm:torch.tensor,box:torch.tensor,device="cpu"):
    """
    :param hm: torch.tensor  shape [128,128]
    :param box: torch.tensor  [x1,y1,x2,y2] 缩减到 hm 大小上
    :return:
    """
    hm = hm.cpu().numpy()
    x1,y1,x2,y2 = box
    h,w = y2-y1,x2-x1
    cx,cy = (x2+x1)/2.,(y2+y1)/2.
    cx,cy = cx.int().item(),cy.int().item()
    h,w = h.item(),w.item()
    radius = gaussian_radius((h,w))
    # radius = math.sqrt(h*w) # 不推荐
    radius = max(1, int(radius))
    # hm = torch.from_numpy(draw_msra_gaussian(hm, (cx,cy), radius)).to(device) # 等价于 drawHeatMapV2
    hm = torch.from_numpy(draw_umich_gaussian(hm, (cx,cy), radius)).to(device)
    return hm

"""参考 centernet 的方式标记 heatmap 并改进"""
def drawHeatMapV2(hm:torch.tensor,box:torch.tensor,device="cpu"):
    """
    :param hm: torch.tensor  shape [128,128]
    :param box: torch.tensor  [x1,y1,x2,y2] 缩减到 hm 大小上
    :return:
    """
    # hm = hm.cpu().numpy()
    x1,y1,x2,y2 = box
    h,w = y2-y1,x2-x1
    cx,cy = (x2+x1)/2.,(y2+y1)/2.
    cx,cy = cx.int().item(),cy.int().item()
    h,w = h.item(),w.item()
    radius = gaussian_radius((h,w))
    # radius = math.sqrt(h*w)# 不推荐
    radius = max(1, int(radius))

    hm = draw_gaussian2(hm,(cy,cx),radius)

    return hm

"""使用fcos 方式标记 heatmap，即框内所有点都做标记"""
def drawHeatMapV3(hm:torch.tensor,box:torch.tensor,device="cpu"):
    """
    :param hm: torch.tensor  shape [128,128]
    :param box: torch.tensor  [x1,y1,x2,y2] 缩减到 hm 大小上 (注： box 已经按照从大到小排序好)
    :return:
    """
    # hm = hm.cpu().numpy()
    x1,y1,x2,y2 = box
    # int_x1 = x1.ceil().int().item()
    # int_y1 = y1.ceil().int().item()
    # int_x2 = x2.floor().int().item()
    # int_y2 = y2.floor().int().item()
    # h,w = y2-y1,x2-x1
    # h,w = h.floor().int().item(),w.floor().int().item()
    int_x1 = x1.floor().int().item()  # +w//4
    int_y1 = y1.floor().int().item()  # +h//4
    int_x2 = x2.ceil().int().item()  # -w//4
    int_y2 = y2.ceil().int().item()  # -h//4

    # 中心点
    cx = (x1+x2)/2.
    cy = (y1+y2)/2.
    cx, cy = cx.int().item(), cy.int().item()

    fh,fw = hm.shape
    for y,x in product(range(int_y1,int_y2),range(int_x1,int_x2)):
        if x <= 0 or y <= 0 or x >= fw or y >= fh: continue
        l = x - x1
        t = y - y1
        r = x2 - x
        b = y2 - y

        if l <= 0 or t <= 0 or r <= 0 or b <= 0: continue

        if x==cx and y==cy:
            centerness=1
        else:
            # centerness = math.sqrt((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))
            centerness = (min(l, r) / max(l, r)) * (min(t, b) / max(t, b))
            # centerness *= np.exp(centerness-1)
        hm[y,x] = centerness

    return hm

def drawHeatMapV0(hm:torch.tensor,box:torch.tensor,device="cpu",thred_radius=1):
    """
    :param hm: torch.tensor  shape [128,128]
    :param box: torch.tensor  [x1,y1,x2,y2] 缩减到 hm 大小上
    :return:
    """
    hm = hm.cpu().numpy()
    x1,y1,x2,y2 = box
    h,w = y2-y1,x2-x1
    cx,cy = (x2+x1)/2.,(y2+y1)/2.
    cx,cy = cx.int().item(),cy.int().item()
    h,w = h.item(),w.item()
    radius = int(gaussian_radius((h,w)))
    # radius = math.sqrt(h*w) # 不推荐
    # radius = max(1, radius)
    if radius > thred_radius:
        hm = torch.from_numpy(draw_umich_gaussian(hm, (cx,cy), radius)).to(device)
    else:
        radius = thred_radius
        hm = torch.from_numpy(draw_msra_gaussian(hm, (cx,cy), radius)).to(device) # 等价于 drawHeatMapV2
    return hm

"""取综合结果
# 推荐使用
"""
def drawHeatMap(hm:torch.tensor,box:torch.tensor,device="cpu",thred_radius=1,dosegm=False):
    """
   :param hm: torch.tensor  shape [128,128]
   :param box: torch.tensor  [x1,y1,x2,y2] 缩减到 hm 大小上 (注： box 已经按照从大到小排序好)
   :return:
   """
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    h, w = h.item(), w.item()
    radius = int(gaussian_radius((h, w)))

    if dosegm:
        if radius <= thred_radius:
            return drawHeatMapV3(hm,box,device)
        else:
            return drawHeatMapV1(hm,box,device)
    else:
        return drawHeatMapV1(hm, box, device)