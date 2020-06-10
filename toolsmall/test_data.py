from data.datasets import PennFudanDataset,PascalVOCDataset
from data.augment import bboxAug
from tools.visual.vis import vis_rect

import numpy as np
import random
import cv2,os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms

def collate_fn(batch_data):
    data_list = []
    target_list = []
    for data,target in batch_data:
        data_list.append(data)
        target_list.append(target)

    return data_list,target_list


def test_datasets():
    # """
    root = r"/media/wucong/225A6D42D4FA828F1/datas/PennFudanPed"
    classes = ["person"]
    typeOfData = "PennFudanDataset"
    """
    root = "/media/wucong/225A6D42D4FA828F1/datas/voc/VOCdevkit/"
    classes = ["aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike",
               "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]
    typeOfData = "PascalVOCDataset"

    # """
    seed = 100
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(seed)
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}

    if typeOfData == "PennFudanDataset":
        Data = PennFudanDataset
    elif typeOfData == "PascalVOCDataset":
        Data = PascalVOCDataset
    else:
        Data = None

    train_transforms = bboxAug.Compose([
               # bboxAug.RandomChoice(),
               # bboxAug.RandomHorizontalFlip(),
               # bboxAug.RandomBrightness(),
               # bboxAug.RandomBlur(),
               # bboxAug.RandomSaturation(),
               # bboxAug.RandomHue(),
               # bboxAug.RandomRotate(angle=5),
               # bboxAug.RandomTranslate(),
               # bboxAug.Augment(False),
               # bboxAug.Pad(), bboxAug.Resize((416,416), False),
               # bboxAug.ResizeMinMax(800,1333),
               bboxAug.ToTensor(), # PIL --> tensor
               # bboxAug.Normalize() # tensor --> tensor
           ])

    dataset = Data(root, 2012, transforms=train_transforms, classes=classes, useMosaic=True)

    data_loader = DataLoader(dataset, batch_size=2, shuffle=False,collate_fn=collate_fn, **kwargs)

    for datas,targets in data_loader:
        for data,target in zip(datas,targets):
            # from c,h,w ->h,w,c
            data = data.permute(1,2,0)
            # to uint8
            data = torch.clamp(data*255,0,255).to("cpu").numpy().astype(np.uint8)

            # to BGR
            # data = data[...,::-1]
            data = cv2.cvtColor(data,cv2.COLOR_RGB2BGR)

            boxes = target["boxes"].to("cpu").numpy().astype(np.int)
            labels = target["labels"].to("cpu").numpy()
            for box,label in zip(boxes,labels):
                data = vis_rect(data,box,str(label),0.5,label)

            cv2.imshow("test", data)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__=="__main__":
    test_datasets()