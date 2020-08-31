"""
统计gt_boxes分布,设计合理的先验anchor,
使用聚类方法找到合适的anchor宽高
"""
from tqdm import tqdm
import torch
from torchvision.datasets import FakeData
import torch.utils.data as data
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

from toolsmall.tools.other import x1y1x2y22xywh,box_iou
from toolsmall.network.generate_anchors import getAnchorsV3

"""统计ground truth boxes的width与height分布"""
def get_wh(data_loader:data.DataLoader):
    wh = []
    for images, targets in tqdm(data_loader):
        bs = len(images)
        for idx in range(bs):
            boxes = targets[idx]["boxes"]
            # labels = targets[idx]["labels"]
            gt_boxes = x1y1x2y22xywh(boxes)
            # 只统计宽高分布
            wh.extend(gt_boxes[:, 2:4])

    wh = torch.stack(wh, 0).numpy()

    return wh

"""使用聚类方法统计，wh的中心，把它当成先验anchor的w，h"""
def static_anchor(wh:np.array,n_clusters=9,random_state = 100,vision=True):
    X = wh
    # Incorrect number of clusters
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    y_pred = model.fit_predict(X)
    centers = model.cluster_centers_
    print(centers)  # 中心点的坐标位置
    if not vision:
        return centers

    # plt.subplot(221)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("Incorrect Number of Blobs")
    plt.xlabel("w")
    plt.ylabel("h")
    # plt.savefig("gt_boxes.jpg")

    plt.scatter(centers[:, 0], centers[:, 1], c="red") # h画出中心点

    plt.show()


def get_confusion_matrix(data_loader:data.DataLoader,num_classes=2,stride=16,iouthresd=0.5,device="cpu",anchors_wh=None):
    """根据IOU 计算混淆矩"""
    confusion_matrix = np.zeros([num_classes, num_classes]) # 0 对应背景
    for images, targets in tqdm(data_loader):
        bs = len(images)
        for idx in range(bs):
            gt_boxes = targets[idx]["boxes"].to(device)/stride  # x1y1x2y2
            labels = targets[idx]["labels"].to(device) # 已经包括背景
            img = images[idx]
            h,w = img.shape[-2:]
            preds_boxes = torch.tensor(getAnchorsV3([h//stride,w//stride],stride,anchors_wh)[0],dtype=torch.float32,device=device)
            ious = box_iou(preds_boxes,gt_boxes)
            # 找到与gt_box 的IOU值最大的预测box
            ious_value, ious_index = ious.max(0)

            # 在判断最大值是否超过0.5
            for idx, (iou, index) in enumerate(zip(ious_value, ious_index)):
                if iou < iouthresd:  # 被分成背景
                    # 真实类别
                    confusion_matrix[labels[idx], 0] += 1
                else:
                    # 如果类别一致则正确分类，否则错误分类
                    confusion_matrix[labels[idx], labels[idx]] += 1

    return confusion_matrix


def get_confusion_matrixV2(data_loader:data.DataLoader,num_classes=2,stride=16,iouthresds=[0.5,0.7,0.9],device="cpu",anchors_wh=None):
    """根据IOU 计算混淆矩"""
    # confusion_matrix = np.zeros([num_classes, num_classes]) # 0 对应背景
    confusion_matrix = {str(iou):np.zeros([num_classes, num_classes]) for iou in iouthresds}

    for images, targets in tqdm(data_loader):
        bs = len(images)
        for idx in range(bs):
            gt_boxes = targets[idx]["boxes"].to(device)/stride  # x1y1x2y2
            labels = targets[idx]["labels"].to(device) # 已经包括背景
            img = images[idx]
            h,w = img.shape[-2:]
            preds_boxes = torch.tensor(getAnchorsV3([h//stride,w//stride],stride,anchors_wh)[0],dtype=torch.float32,device=device)
            ious = box_iou(preds_boxes,gt_boxes)
            # 找到与gt_box 的IOU值最大的预测box
            ious_value, ious_index = ious.max(0)

            # 在判断最大值是否超过0.5
            for idx, (iou, index) in enumerate(zip(ious_value, ious_index)):
                for iouthresd in iouthresds:
                    if iou < iouthresd:  # 被分成背景
                        # 真实类别
                        confusion_matrix[str(iouthresd)][labels[idx], 0] += 1
                    else:
                        # 如果类别一致则正确分类，否则错误分类
                        confusion_matrix[str(iouthresd)][labels[idx], labels[idx]] += 1

    return confusion_matrix

"""for evalute"""
def get_confusion_matrixV3(data_loader:data.DataLoader,preds=[],num_classes=2,iouthresds=[0.5,0.7,0.9],device="cpu"): # batchsize=1
    """根据IOU 计算混淆矩"""
    # confusion_matrix = np.zeros([num_classes, num_classes]) # 0 对应背景
    confusion_matrix = {str(iou):np.zeros([num_classes, num_classes]) for iou in iouthresds}
    for idx,(images, targets) in tqdm(enumerate(data_loader)):
        gt_boxes = targets[0]["boxes"].to(device)  # x1y1x2y2
        labels = targets[0]["labels"].to(device) # 已经包括背景
        pred = preds[idx]
        preds_boxes = pred["boxes"]
        preds_labels = pred["labels"]

        if len(preds_boxes)==0:
            for iouthresd in iouthresds:
                for label in labels:
                    confusion_matrix[str(iouthresd)][int(label.item()), 0] += 1
        else:
            ious = box_iou(preds_boxes,gt_boxes)
            # 找到与gt_box 的IOU值最大的预测box
            ious_value, ious_index = ious.max(0)

            # 在判断最大值是否超过0.5
            for idx, (iou, index) in enumerate(zip(ious_value, ious_index)):
                for iouthresd in iouthresds:
                    if iou < iouthresd:  # 被分成背景
                        # 真实类别
                        confusion_matrix[str(iouthresd)][labels[idx], 0] += 1
                    else:
                        # 如果类别一致则正确分类，否则错误分类
                        confusion_matrix[str(iouthresd)][labels[idx], preds_labels[index]] += 1

    return confusion_matrix

# 计算precision，recall，f1_score
def cal_confusion_matrix(matrix, cate_list=[], save_path=None):  # "./test.csv"
    # matrix=metrics.confusion_matrix(y_true,y_pred)
    h, w = matrix.shape
    new_matrix = np.zeros([h + 1, w + 2], np.float32)
    new_matrix[:h, :w] = matrix
    for row, item in enumerate(matrix):
        new_matrix[row, -2] = item[row] / np.sum(item)  # 召回率 recall

    for row, item in enumerate(matrix.T):
        new_matrix[-1, row] = item[row] / np.sum(item)  # 准确率 precision

    for i, (P, R) in enumerate(zip(new_matrix[-1, :], new_matrix[:, -2])):
        new_matrix[i, -1] = 0 if P + R == 0 else 2 * P * R / (P + R)  # 计算 F1

    # df=pd.DataFrame(new_matrix,index=[*cate_list,"precision"],columns=[*cate_list,"recall","F1"])
    # cate_list.append("backgrund")
    df = pd.DataFrame(matrix, index=cate_list, columns=cate_list)
    # df["score"]=None # 增加一列score
    df["precision"] = new_matrix[-1, :-2]
    df["recall"] = new_matrix[:-1, -2]
    df["F1"] = new_matrix[:-1, -1]

    if save_path == None:
        # print(df)
        sys.stdout.write("%s\n" % (str(df)))
    else:
        # df.to_csv("./test.csv",index=False)
        df.to_csv(os.path.join(save_path, "evaluate.csv"))

if __name__ == "__main__":
    from toolsmall.data.data import Datas

    """
    classes = ["__background__", "person"]
    preddataPath = None
    testdataPath = None
    traindataPath = r"/media/wucong/225A6D42D4FA828F1/datas/PennFudanPed"
    typeOfData = "PennFudanDataset"

    """
    classes = ["__background__","date", "fig", "hazelnut"]
    preddataPath = None
    testdataPath = None
    traindataPath = "/media/wucong/225A6D42D4FA828F1/datas/data"
    typeOfData = "FruitsNutsDataset"
    # """

    data = Datas(True, traindataPath, testdataPath, preddataPath, typeOfData, classes, min_size=600, max_size=1000,
                 batch_size=1,useImgaug=False)

    wh = get_wh(data.train_loader)
    anchors_wh = static_anchor(wh,5,vision=False)

    # confusion_matrix = get_confusion_matrix(data.train_loader,len(classes),stride=8,iouthresd=0.7,anchors_wh=anchors_wh)
    # cal_confusion_matrix(confusion_matrix, classes)

    confusion_matrixs = get_confusion_matrixV2(data.train_loader,len(classes),stride=8,anchors_wh=anchors_wh)

    for k,v in confusion_matrixs.items():
        print("\n",k,"\n")
        cal_confusion_matrix(v,classes)
