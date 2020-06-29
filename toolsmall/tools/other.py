flag=False
try:
    from torchvision.ops.boxes import batched_nms as nms
    flag=True
except:
    from .nms.nms import nms2
from .visual.vis import vis_rect,vis_keypoints2
import torch
from torch import nn

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


        # 2.类别一样的按nms过滤，如果Iou大于nms_thres,保留分数最大的,否则都保留
        # 按阈值过滤
        scores = prediction["scores"][ms]
        labels = prediction["labels"][ms]
        boxes = prediction["boxes"][ms]
        if "keypoints" in prediction:
            keypoints = prediction["keypoints"][ms]
        if flag:
            keep = nms(boxes, scores,labels,nms_thres)
            last_scores.extend(scores[keep])
            last_labels.extend(labels[keep])
            last_boxes.extend(boxes[keep])
            if "keypoints" in prediction:
                last_keypoints.extend(keypoints[keep])

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

        if "keypoints" in prediction:
            return {"scores": last_scores, "labels": last_labels, "boxes": last_boxes,"keypoints":last_keypoints}

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

# ----------------------------------------------------------------

if __name__=="__main__":
    pass