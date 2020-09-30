flag=False
try:
    from torchvision.ops.boxes import batched_nms as nms
    flag=True
except:
    from .nms.nms import nms2
try:
    from visual.vis import vis_rect,vis_keypoints2,drawMask
except:
    from .visual.vis import vis_rect,vis_keypoints2,drawMask
import torch
from torch import nn
import cv2
from torchvision.ops import roi_pool,roi_align
from skimage.transform import resize
import numpy as np
from itertools import product
from torchvision.ops import misc as misc_nn_ops
import matplotlib.pyplot as plt
import math

def apply_nms(prediction,conf_thres=0.3,nms_thres=0.4,filter_labels=[]):
    """
    Parameters
    ----------
    prediction:
            {"boxes":Tensor[N,4],"labels":Tensor[N,],"scores":Tensor[N,]}

    Returns:
    -------
           {"boxes":Tensor[N,4],"labels":Tensor[N,],"scores":Tensor[N,]}

    """
    ms = prediction["scores"] > conf_thres
    if torch.sum(ms) == 0:
        return None
    else:
        last_scores = []
        last_labels = []
        last_boxes = []
        if "keypoints" in prediction:
            last_keypoints = []

        if "masks" in prediction:
            last_masks = []


        # 2.类别一样的按nms过滤，如果Iou大于nms_thres,保留分数最大的,否则都保留
        # 按阈值过滤
        scores = prediction["scores"][ms]
        labels = prediction["labels"][ms]
        boxes = prediction["boxes"][ms]
        if "keypoints" in prediction:
            keypoints = prediction["keypoints"][ms]
        if "masks" in prediction:
            masks = prediction["masks"][ms]

        if flag:
            keep = nms(boxes, scores,labels,nms_thres)
            last_scores.extend(scores[keep])
            last_labels.extend(labels[keep])
            last_boxes.extend(boxes[keep])
            if "keypoints" in prediction:
                last_keypoints.extend(keypoints[keep])

            if "masks" in prediction:
                last_masks.extend(masks[keep])

        else:
            unique_labels = labels.unique()
            for c in unique_labels:
                if c in filter_labels: continue

                # Get the detections with the particular class
                temp = labels == c
                _scores = scores[temp]
                _labels = labels[temp]
                _boxes = boxes[temp]
                if len(_labels) > 1:
                    keep = nms2(_boxes, _scores, nms_thres)
                    last_scores.extend(_scores[keep])
                    last_labels.extend(_labels[keep])
                    last_boxes.extend(_boxes[keep])

                else:
                    last_scores.extend(_scores)
                    last_labels.extend(_labels)
                    last_boxes.extend(_boxes)

        if len(last_labels) == 0:
            return None

        # resize 到原图上
        h_ori, w_ori = prediction["original_size"]
        h_re, w_re = prediction["resize"]
        h_ori = h_ori.float()
        w_ori = w_ori.float()

        # to pad图上
        if h_ori > w_ori:
            h_scale = h_ori / h_re
            w_scale = h_ori / w_re
            # 去除pad部分
            diff = h_ori - w_ori
            for i in range(len(last_boxes)):
                last_boxes[i][[0, 2]] *= w_scale
                last_boxes[i][[1, 3]] *= h_scale

                last_boxes[i][0] -= diff // 2
                last_boxes[i][2] -= diff - diff // 2

                if "keypoints" in prediction:
                    last_keypoints[i][:,0] *= w_scale
                    last_keypoints[i][:,1] *= h_scale
                    last_keypoints[i][:, 0] -= diff // 2

                if "masks" in prediction:
                    last_masks[i] = cv2.resize(last_masks[i].cpu().numpy(),(h_ori,h_ori),interpolation=cv2.INTER_NEAREST)
                    start = int(diff.item() // 2)
                    end = int(diff.item()-diff.item() // 2)
                    last_masks[i] = torch.from_numpy(last_masks[i][:, start:-end])

        else:
            h_scale = w_ori / h_re
            w_scale = w_ori / w_re
            diff = w_ori - h_ori
            for i in range(len(last_boxes)):
                last_boxes[i][[0, 2]] *= w_scale
                last_boxes[i][[1, 3]] *= h_scale

                last_boxes[i][1] -= diff // 2
                last_boxes[i][3] -= diff - diff // 2

                if "keypoints" in prediction:
                    last_keypoints[i][:,0] *= w_scale
                    last_keypoints[i][:,1] *= h_scale
                    last_keypoints[i][:, 1] -= diff // 2
                if "masks" in prediction:
                    last_masks[i] = cv2.resize(last_masks[i].cpu().numpy(),(w_ori,w_ori),interpolation=cv2.INTER_NEAREST)
                    start = int(diff.item() // 2)
                    end = int(diff.item() - diff.item() // 2)
                    last_masks[i] = torch.from_numpy(last_masks[i][start:-end,:])

        if "keypoints" in prediction:
            return {"scores": last_scores, "labels": last_labels, "boxes": last_boxes,"keypoints":last_keypoints}

        if "masks" in prediction:
            return {"scores": last_scores, "labels": last_labels, "boxes": last_boxes,"masks":last_masks}

        return {"scores": last_scores, "labels": last_labels, "boxes": last_boxes}

def draw_rect(image, pred,classes=[],inside=False):
    """
    Parameters
    ----------
    image:
            np.array[h,w,3] ,0~255
    pred:
            {"boxes":Tensor[N,4],"labels":Tensor[N,],"scores":Tensor[N,]}
    classes:
            ["bicycle", "bus", "car", "motorbike", "person"] 注意默认不包含背景的

    Returns:
    -------
    image:
        np.array[h,w,3] ,0~255

    """

    labels = pred["labels"]
    bboxs = pred["boxes"]
    scores = pred["scores"]

    for label, bbox, score in zip(labels, bboxs, scores):
        label = label.cpu().numpy()
        bbox = bbox.cpu().numpy()  # .astype(np.int16)
        score = score.cpu().numpy()
        if classes:
            class_str = "%s:%.3f" % (classes[int(label)], score)  # 跳过背景
        else:
            class_str = "%s:%.3f" % (int(label), score)
        pos = list(map(int, bbox))

        image = vis_rect(image, pos, class_str, 0.5, int(label),inside=inside)
    return image

def draw_rect_mask(image, pred,classes=[],inside=False):
    """
    Parameters
    ----------
    image:
            np.array[h,w,3] ,0~255
    pred:
            {"boxes":Tensor[N,4],"labels":Tensor[N,],"scores":Tensor[N,]}
    classes:
            ["__background__","bicycle", "bus", "car", "motorbike", "person"] 注意包含背景的

    Returns:
    -------
    image:
        np.array[h,w,3] ,0~255

    """

    labels = pred["labels"]
    bboxs = pred["boxes"]
    scores = pred["scores"]
    if "masks" in pred:
        masks = pred["masks"]
    if "keypoints" in pred:
        keypoints = pred["keypoints"]


    for idx,(label, bbox, score) in enumerate(zip(labels, bboxs, scores)):
        label = label.cpu().numpy()
        bbox = bbox.cpu().numpy()  # .astype(np.int16)
        score = score.cpu().numpy()

        if classes:
            class_str = "%s:%.3f" % (classes[int(label)], score)  # 跳过背景
        else:
            class_str = "%s:%.3f" % (int(label), score)
        pos = list(map(int, bbox))

        image = vis_rect(image, pos, class_str, 0.5, int(label),inside=inside,useMask=False)
        if "masks" in pred:
            mask = masks[idx]
            if mask.max() > 0:
                image = drawMask(image, mask.cpu().numpy(), label, alpha=0.7)

    if "keypoints" in pred:
        image = vis_keypoints2(image,keypoints.cpu().numpy().transpose([0,2,1]),1)

    return image

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

def weights_init_fpn(model):
    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # if m.bias is not None:
            #     nn.init.zeros_(m.bias)
            # c2_msra_fill(m)
            c2_xavier_fill(m)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def weights_init_rpn(model):
    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def weights_init(model):
    """
    :param model: nn.Model
    :return:

    :example
        m = nn.Sequential(
            nn.Conv2d(3,32,3,2,1),
            nn.BatchNorm2d(32),
            nn.ReLU())

        m.apply(weights_init)
    """
    mode = 'fan_in';slope = 0.1

    # for m in self.modules():
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=slope, mode=mode)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

# -------------------mask---------------------------------------------
def proposal2GTMask(gt_mask,proposal,outsize=(14,14)):
    """根据生成的区域建议框，到gt_mask截取相应的部分再resize到指定大小，用于maskhead 计算loss
        gt_mask: [1,m,1000,600]
        proposal:[[0,0,1000,600]] # 大小缩放到原始输入大小
    """
    target_mask = roi_align(gt_mask, proposal, outsize, 1.0)
    return target_mask.byte().float()

# https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py#L560
def unmold_mask(mask, bbox, image_shape,origin_shape=None):
    """Converts a mask generated by the neural network to a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.
    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    mask = mask >= threshold

    x1, y1, x2, y2 = bbox
    mask = misc_nn_ops.interpolate(mask[None,None].float(), size=(y2 - y1, x2 - x1), mode="nearest")[0,0].byte()

    # Put the mask in the right location.
    full_mask = torch.zeros(image_shape, dtype=torch.bool,device=mask.device)
    full_mask[y1:y2, x1:x2] = mask

    if origin_shape is not None:
        full_mask = misc_nn_ops.interpolate(full_mask[None,None].float(), size=origin_shape, mode="nearest")[0,0].byte()

    return full_mask


def proposal2GTMaskV2(gt_mask,proposal,outsize=(14,14)):
    """根据生成的区域建议框，到gt_mask截取相应的部分再resize到指定大小，用于maskhead 计算loss
        gt_mask: [1,m,1000,600]
        proposal:[[0,0,1000,600]] # 大小缩放到原始输入大小
    """
    target_mask = roi_align(gt_mask, proposal, outsize, 1.0)
    return target_mask#.byte().float()

# https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py#L560
def unmold_maskV2(mask, bbox, image_shape,origin_shape=None):
    """Converts a mask generated by the neural network to a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.
    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    # mask = mask >= threshold

    x1, y1, x2, y2 = bbox
    mask = misc_nn_ops.interpolate(mask[None,None].float(), size=(y2 - y1, x2 - x1), mode="bilinear")[0,0]#.byte()

    # Put the mask in the right location.
    full_mask = torch.zeros(image_shape, dtype=torch.float32,device=mask.device)
    full_mask[y1:y2, x1:x2] = mask

    if origin_shape is not None:
        full_mask = misc_nn_ops.interpolate(full_mask[None,None].float(), size=origin_shape, mode="bilinear")[0,0]#.byte()

    return full_mask >= threshold


# -------------------keypoint--------------------------------------------------
# from torchvision.models.detection.roi_heads import keypoints_to_heatmap
def draw_gaussian2(heatmap,center,sigma=1.0):
    # center: y,x
    h,w = heatmap.shape
    X, Y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    yx = np.concatenate((Y[..., None],X[..., None]), -1).astype(np.float32)
    yx = torch.from_numpy(yx).to(heatmap.device)
    ht = torch.exp(-((yx[...,0] - center[0]) ** 2 + (yx[...,1] - center[1]) ** 2) / (2 * sigma ** 2))
    # ht *= (ht>0)
    # heatmap = ht
    heatmap = heatmap*(heatmap-ht>=0)+ht*(heatmap-ht<0)

    # for i, j in product(range(h), range(w)):
    #     heatmap[i, j] = max(heatmap[i, j],torch.exp(-((i - center[0]) ** 2 + (j - center[1]) ** 2) / (2 * sigma ** 2)))

    return heatmap

def keypoints_to_heatmap(keypoints, rois, heatmap_size=56,surrounding=False):
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    heatmap = torch.zeros([*valid.shape,heatmap_size,heatmap_size],dtype=torch.float32,device=keypoints.device)

    bs,nums_keypoints = valid.shape
    for i in range(bs):
        for j in range(nums_keypoints):
            if valid[i,j]>0:
                if surrounding: # 考虑周围点
                    heatmap[i,j] = draw_gaussian2(heatmap[i,j],(y[i,j],x[i,j]),1.0)
                else:
                    heatmap[i,j,y[i,j],x[i,j]] = valid[i,j] # 只考虑一个点，不考虑周围点


    return heatmap

"""（按mask方式做，先把keypoint转成对应的mask）这个会转换后出现多个点,不推荐"""
def keypoints_to_heatmapV2(keypoints_mask, proposal, heatmap_size=56):
    """
    img = np.zeros([448,448],np.uint8)
    img[224,224] = 1

    img = cv2.resize(img,(56,56),interpolation=cv2.INTER_NEAREST)
    img = img==img.max()
    """
    target_mask = roi_align(keypoints_mask, proposal, heatmap_size, 1.0)

    return target_mask#.byte().float()

# 从预测出的heatmap反算出keypoint在原图的位置
def heatmap_to_keypoints(heatmap, rois, heatmap_size=56,threds=0.5):
    bs,nums_keypoints = heatmap.shape[:2]
    heatmap = heatmap.view(bs,nums_keypoints,-1)
    max_v,index = heatmap.max(-1)
    y = index//heatmap_size
    x = index%heatmap_size
    v = torch.ones_like(x)
    keypoints = torch.stack((x,y,v),-1)
    keypoints = keypoints*((max_v>=threds).float().unsqueeze(-1)) # [16,17,3]

    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x = x / scale_x + offset_x
    y = y / scale_y + offset_y

    keypoints[..., 0] = x
    keypoints[...,1] = y

    return keypoints

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
    int_x1 = x1.ceil().int().item()
    int_y1 = y1.ceil().int().item()
    int_x2 = x2.floor().int().item()
    int_y2 = y2.floor().int().item()
    for y,x in product(range(int_y1,int_y2),range(int_x1,int_x2)):
        l = x - x1
        t = y - y1
        r = x2 - x
        b = y2 - y
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

# ------------------------------------------------------
def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _nms(heat, kernel=3):
    """
    :param heat: torch.tensor [bs,c,h,w]
    :param kernel:
    :return:
    """
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    """
    :param scores:  torch.tensor [bs,c,h,w]
    :param K:
    :return:
    """
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K) # 每个类别取前K个 shape : [batch,cat,K]

    # topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float() # [batch,cat,K]
    topk_xs = (topk_inds % width).int().float() # [batch,cat,K]

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K) # 按分数排序取 前K个 shape：[batch,K]
    topk_clses = (topk_ind / K).int() # 计算对应的类别 id
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def _topk2(scores, K=40):
    """
    :param scores:  torch.tensor [bs,c,h,w] , batch=1
    :param K:
    :return:
    """
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K) # 每个类别取前K个 shape : [batch,cat,K]

    # topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float() # [batch,cat,K]
    topk_xs = (topk_inds % width).int().float() # [batch,cat,K]

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K) # 按分数排序取 前K个 shape：[batch,K]
    topk_clses = (topk_ind / K).int() # 计算对应的类别 id
    topk_k = (topk_ind % K).int()

    _topk_inds = []
    _topk_ys = []
    _topk_xs = []
    for c,k in zip(topk_clses,topk_k):
        _topk_inds.append(topk_inds[:,c,k])
        _topk_ys.append(topk_ys[:,c,k])
        _topk_xs.append(topk_xs[:,c,k])

    topk_inds = torch.stack(_topk_inds,-1)
    topk_ys = torch.stack(_topk_ys,-1)
    topk_xs = torch.stack(_topk_xs,-1)

    # topk_inds = _gather_feat(
    #     topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    # topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    # topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _topk3(scores,regBoxes, K=40):
    """
    :param scores:  torch.tensor [h,w,c]  包括背景
    :param regBoxes:  torch.tensor [h,w,c,4]
    :param K:
    :return:
    """
    height, width,cat = scores.size()

    regBoxes = regBoxes.contiguous().view(-1,4)

    topk_scores, topk_inds = torch.topk(scores.view(-1,cat), K,0) # 每个类别取前K个 shape : [K,cat]
    regBoxes = regBoxes[topk_inds] # [K,cat,4]

    # topk_inds = topk_inds % (height * width)
    # topk_ys = (topk_inds / width).int().float() # [K,cat]
    topk_ys = (topk_inds // width).int().float() # [K,cat]
    topk_xs = (topk_inds % width).int().float() # [K,cat]

    topk_score, topk_ind = torch.topk(topk_scores.view(-1), K) # 按分数排序取 前K个 shape：[batch,K]
    # topk_k = (topk_ind / cat).int()
    topk_k = (topk_ind // cat).int()
    topk_clses = (topk_ind % cat).int()# 计算对应的类别 id

    _topk_inds = []
    _topk_ys = []
    _topk_xs = []
    _regBoxes = []
    for c,k in zip(topk_clses,topk_k):
        _topk_inds.append(topk_inds[k,c])
        _topk_ys.append(topk_ys[k,c])
        _topk_xs.append(topk_xs[k,c])
        _regBoxes.append(regBoxes[k,c])

    topk_inds = torch.stack(_topk_inds,-1)
    topk_ys = torch.stack(_topk_ys,-1)
    topk_xs = torch.stack(_topk_xs,-1)
    regBoxes = torch.stack(_regBoxes,0)

    # topk_inds = _gather_feat(
    #     topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    # topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    # topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, regBoxes, topk_clses, topk_ys, topk_xs


if __name__ == "__main__":
    # scores = torch.rand([128,128,21])
    # regBoxes = torch.rand([128,128,21,4])
    # _topk3(scores,regBoxes)
    # exit(0)

    # heatmap = torch.rand([28,28,10])
    # heatmap2 = torch.rand([28,28,10,4])
    # heatmap2indexV2(heatmap,heatmap2,0.5,True)
    # radius = gaussian_radius((21,32)) # (21,32) 缩放到 heatmap上
    # radius = max(1, int(radius))
    # hm = np.zeros([128,128],np.float32)
    # hm = draw_umich_gaussian(hm,(64,64),radius) # 小
    # hm2 = np.zeros([128, 128], np.float32)
    # hm2 = draw_msra_gaussian(hm2, (64, 64), radius) # 大

    hm = torch.zeros((128,128))
    # draw_msra_gaussian(hm.clone().numpy(),(64,64),1)
    # draw_umich_gaussian(hm.clone().numpy(),(64,64),3)
    # draw_gaussian2(hm,(64,64),1)
    # box = torch.tensor((34,34,94,94),dtype=torch.float32)
    box = torch.tensor((44,44,84,84),dtype=torch.float32)
    # box = torch.tensor((54,54,74,74),dtype=torch.float32)
    # box = torch.tensor((59,59,69,69),dtype=torch.float32)
    # box = torch.tensor((62,62,66,66),dtype=torch.float32)
    hm1 = drawHeatMapV1(hm.clone(),box).cpu().numpy()
    hm2 = drawHeatMapV2(hm.clone(),box).cpu().numpy()
    hm3 = drawHeatMapV3(hm.clone(),box).cpu().numpy()
    hm4 = drawHeatMap(hm.clone(),box).cpu().numpy()
    hm5 = drawHeatMapV0(hm.clone(),box).cpu().numpy()
    wh = box[2:]-box[:2]
    w = wh[0].item()
    h = wh[1].item()
    radius = gaussian_radius((h, w))
    print(radius)

    plt.subplot(1,5,1)
    plt.imshow(hm1,"gray")
    plt.subplot(1, 5, 2)
    plt.imshow(hm2, "gray")
    plt.subplot(1, 5, 3)
    plt.imshow(hm3, "gray")
    plt.subplot(1, 5, 4)
    plt.imshow(hm4, "gray")
    plt.subplot(1, 5, 5)
    plt.imshow(hm5, "gray")
    plt.show()