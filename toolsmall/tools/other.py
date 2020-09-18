flag=False
try:
    from torchvision.ops.boxes import batched_nms as nms
    flag=True
except:
    from .nms.nms import nms2
from .visual.vis import vis_rect,vis_keypoints2,drawMask
import torch
from torch import nn
import cv2
from torchvision.ops import roi_pool,roi_align
from skimage.transform import resize
import numpy as np
from itertools import product

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

def draw_rect(image, pred,classes=[]):
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

        image = vis_rect(image, pos, class_str, 0.5, int(label))
    return image

def draw_rect_mask(image, pred,classes=[]):
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

        image = vis_rect(image, pos, class_str, 0.5, int(label),useMask=False)
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
    return target_mask

# https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py#L560
def unmold_mask(mask, bbox, image_shape,origin_shape=None):
    """Converts a mask generated by the neural network to a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.
    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    x1, y1, x2, y2 = bbox
    mask = resize(mask, (y2 - y1, x2 - x1),mode="constant") # "constant" ， wrap
    # mask = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST) # 直接插值效果不好
    # mask = np.where(mask >= threshold, 1, 0).astype(np.bool)
    mask = mask>threshold

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape, dtype=np.bool)
    full_mask[y1:y2, x1:x2] = mask

    if origin_shape is not None:
        full_mask = resize(full_mask,origin_shape,mode="constant")

    return full_mask

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

    heatmap = torch.zeros([*valid.shape,heatmap_size,heatmap_size],dtype=torch.long,device=keypoints.device)

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

    return target_mask

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





if __name__ == "__main__":
    heatmap = torch.rand([28,28,10])
    heatmap2 = torch.rand([28,28,10,4])
    heatmap2indexV2(heatmap,heatmap2,0.5,True)