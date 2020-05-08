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

def nms2(dets,scores,thresh=0.4):
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

def nms(dets,thresh=0.4):
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
    boxes=torch.as_tensor(boxes)
    keep = nms(boxes[:,:4],boxes[:,-1], thresh=0.7)
    plot_bbox(boxes[[keep]].numpy(), 'r')  # after nms
    # or torch.index_select(boxes,dim=0,index=torch.as_tensor(keep))

    plt.show()