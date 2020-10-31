#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# https://www.cnblogs.com/king-lps/p/9031568.html

Created on Mon May  7 21:45:37 2018

@author: lps
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

def nms(dets,scores,thresh=0.4,nums=50):
    """
    :param dets: [N,4]
    :param scores: [N]
    :param thresh: iou阈值
    :return: index(保留下来的索引值)
    """

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    keep = []

    # scores = dets[:, 4]
    # index = scores.argsort()[::-1] # 按分数从大到小排序其索引,默认是从小到大排序,[::-1]采用倒序排列

    # Sort the detections by maximum objectness confidence
    _, index = torch.sort(scores, descending=True)

    while index.numel() > 1:
        try:
            if len(keep) > nums: break
            i = index[0].item()  # every time the first is the biggst, and add it directly
            # i = index.cpu().numpy()[0]
            keep.append(i)

            x11 = torch.max(x1[i], x1[index[1:]])  # calculate the points of overlap
            y11 = torch.max(y1[i], y1[index[1:]])
            x22 = torch.min(x2[i], x2[index[1:]])
            y22 = torch.min(y2[i], y2[index[1:]])


            # w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
            # h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

            w=(x22 - x11 + 1).clamp(min=0)
            h=(y22 - y11 + 1).clamp(min=0)


            overlaps = w * h

            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

            # idx = np.where(ious.to("cpu").numpy() <= thresh)[0] # 小于阈值bbox被保留，进行下一次迭代
            idx=torch.nonzero(ious <= thresh).squeeze()
            index = index[idx + 1]  # because index start from 1
        except:
            pass

    return keep

def diouNms(dets,scores,thresh=0.4,nums=50):
    """
    :param dets: [N,4]
    :param scores: [N]
    :param thresh: iou阈值
    :return: index(保留下来的索引值)
    """

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    keep = []

    # scores = dets[:, 4]
    # index = scores.argsort()[::-1] # 按分数从大到小排序其索引,默认是从小到大排序,[::-1]采用倒序排列

    # Sort the detections by maximum objectness confidence
    _, index = torch.sort(scores, descending=True)

    while index.numel() > 1:
        try:
            if len(keep)>nums:break
            i = index[0].item()  # every time the first is the biggst, and add it directly
            # i = index.cpu().numpy()[0]
            keep.append(i)

            x11 = torch.max(x1[i], x1[index[1:]])  # calculate the points of overlap
            y11 = torch.max(y1[i], y1[index[1:]])
            x22 = torch.min(x2[i], x2[index[1:]])
            y22 = torch.min(y2[i], y2[index[1:]])

            # """
            _x11 = torch.min(x1[i], x1[index[1:]])  # calculate the points of overlap
            _y11 = torch.min(y1[i], y1[index[1:]])
            _x22 = torch.max(x2[i], x2[index[1:]])
            _y22 = torch.max(y2[i], y2[index[1:]])

            _w = (_x22 - _x11 + 1).clamp(min=0)
            _h = (_y22 - _y11 + 1).clamp(min=0)

            cx = (x2[i]+x1[i])/2
            cy = (y2[i]+y1[i])/2
            _cx = (x2[index[1:]] + x1[index[1:]]) / 2
            _cy = (y2[index[1:]] + y1[index[1:]]) / 2

            d = (cx-_cx)**2+(cy-_cy)**2
            c = _w**2+_h**2

            rdiou = d/c
            # """

            # w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
            # h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

            w=(x22 - x11 + 1).clamp(min=0)
            h=(y22 - y11 + 1).clamp(min=0)

            overlaps = w * h

            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
            ious -= rdiou # Diou

            # idx = np.where(ious.to("cpu").numpy() <= thresh)[0] # 小于阈值bbox被保留，进行下一次迭代
            idx=torch.nonzero(ious <= thresh).squeeze()
            index = index[idx + 1]  # because index start from 1
        except:
            pass

    return keep


def diou_softNms(dets,scores,iou_thresh=0.4,conf_thresh=0.05,nums=50,method=2,sigma=0.5,use_diou=True):
    """
    :param dets: [N,4]
    :param scores: [N]
    :param thresh: iou阈值
    :return: index(保留下来的索引值)
    """
    size = dets.size(0)
    _dets = []
    _scores = []
    while len(_scores)<=size and len(_scores)<=nums:
        # Sort the detections by maximum objectness confidence
        _, index = torch.sort(scores, descending=True)
        i = index[0].item()
        # keep.append(i)
        _dets.append(dets[index[0]])
        _scores.append(scores[index[0]])

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (y2 - y1 + 1) * (x2 - x1 + 1)

        x11 = torch.max(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = torch.max(y1[i], y1[index[1:]])
        x22 = torch.min(x2[i], x2[index[1:]])
        y22 = torch.min(y2[i], y2[index[1:]])

        if use_diou:
            _x11 = torch.min(x1[i], x1[index[1:]])  # calculate the points of overlap
            _y11 = torch.min(y1[i], y1[index[1:]])
            _x22 = torch.max(x2[i], x2[index[1:]])
            _y22 = torch.max(y2[i], y2[index[1:]])

            _w = (_x22 - _x11 + 1).clamp(min=0)
            _h = (_y22 - _y11 + 1).clamp(min=0)

            cx = (x2[i] + x1[i]) / 2
            cy = (y2[i] + y1[i]) / 2
            _cx = (x2[index[1:]] + x1[index[1:]]) / 2
            _cy = (y2[index[1:]] + y1[index[1:]]) / 2

            d = (cx - _cx) ** 2 + (cy - _cy) ** 2
            c = _w ** 2 + _h ** 2

            rdiou = d / c


        # w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        # h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        w = (x22 - x11 + 1).clamp(min=0)
        h = (y22 - y11 + 1).clamp(min=0)

        overlaps = w * h

        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        if use_diou:ious -= rdiou  # Diou
        weight = torch.ones_like(ious)

        if method == 1:  # linear
            weight[(ious-iou_thresh)>0] = 1-ious[(ious-iou_thresh)>0]
        elif method == 2: # gaussian
            weight = torch.exp(-(ious * ious) / sigma)
        else:# original NMS
            weight[(ious - iou_thresh) > 0] = 0

        scores = scores[index[1:]]*weight
        dets = dets[index[1:]]

        if len(scores)==0:break

    if len(_dets)>0:
        _dets,_scores=torch.stack(_dets, 0), torch.stack(_scores, 0)
        keep = _scores>=conf_thresh
        if keep.sum()>0:
            _dets, _scores = _dets[keep],_scores[keep]
        else:
            _dets = []
            _scores = []
    return _dets,_scores


"""
按每个类别 做 soft_nms
"""
def soft_nms(boxes: np.ndarray, sigma: float = 0.5, Nt: float = 0.3, threshold: float = 0.001, method: int = 1):
    """boxes:[bs,5]  [x1,y1,x2,y2,scores]"""
    N = boxes.shape[0]
    # pos = 0
    # maxscore = 0
    # maxpos = 0

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i, 0] = boxes[maxpos, 0]
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        boxes[i, 4] = boxes[maxpos, 4]

        # swap ith box with position of max box
        boxes[maxpos, 0] = tx1
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        boxes[maxpos, 4] = ts

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # iou between max box and detection box

                    if method == 1:  # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov) / sigma)
                    else:  # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight * boxes[pos, 4]

                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos, 0] = boxes[N - 1, 0]
                        boxes[pos, 1] = boxes[N - 1, 1]
                        boxes[pos, 2] = boxes[N - 1, 2]
                        boxes[pos, 3] = boxes[N - 1, 3]
                        boxes[pos, 4] = boxes[N - 1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    return keep


def nmsV2(dets,thresh=0.4):
    """
    :param dets: [N,5]
    :param thresh: iou阈值
    :return: index(保留下来的索引值)
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    keep = []

    # scores = dets[:, 4]
    # index = scores.argsort()[::-1] # 按分数从大到小排序其索引,默认是从小到大排序,[::-1]采用倒序排列

    # Sort the detections by maximum objectness confidence
    _, index = torch.sort(scores, descending=True)

    while index.numel() > 1:
        try:
            i = index[0].item()  # every time the first is the biggst, and add it directly
            # i = index.cpu().numpy()[0]
            keep.append(i)

            x11 = torch.max(x1[i], x1[index[1:]])  # calculate the points of overlap
            y11 = torch.max(y1[i], y1[index[1:]])
            x22 = torch.min(x2[i], x2[index[1:]])
            y22 = torch.min(y2[i], y2[index[1:]])


            # w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
            # h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

            w=(x22 - x11 + 1).clamp(min=0)
            h=(y22 - y11 + 1).clamp(min=0)

            overlaps = w * h

            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

            # idx = np.where(ious.to("cpu").numpy() <= thresh)[0] # 小于阈值bbox被保留，进行下一次迭代
            idx=torch.nonzero(ious <= thresh).squeeze()
            index = index[idx + 1]  # because index start from 1
        except:
            pass

    return keep

def plot_bbox(dets, c='k'):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    plt.plot([x1, x2], [y1, y1], c)
    plt.plot([x1, x1], [y1, y2], c)
    plt.plot([x1, x2], [y2, y2], c)
    plt.plot([x2, x2], [y1, y2], c)
    plt.title("after nms")


if __name__=="__main__":
    boxes = np.array([[100, 100, 210, 210, 0.72],
                      [250, 250, 420, 420, 0.8],
                      [220, 220, 320, 330, 0.92],
                      [100, 100, 210, 210, 0.72],
                      [230, 240, 325, 330, 0.81],
                      [220, 230, 315, 340, 0.9]])

    plot_bbox(boxes, 'k')  # before nms

    # keep = py_cpu_nms(boxes, thresh=0.7)
    # keep = soft_nms(boxes, Nt=0.7,threshold=0.3,method=1)
    boxes=torch.as_tensor(boxes)
    dets,scores = diou_softNms(boxes[:,:4],boxes[:,-1], iou_thresh=0.4,conf_thresh=0.5)
    plot_bbox(dets.numpy(), 'r')  # after nms
    # or torch.index_select(boxes,dim=0,index=torch.as_tensor(keep))

    plt.show()