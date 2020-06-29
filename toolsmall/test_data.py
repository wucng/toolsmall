from data.datasets import PennFudanDataset,PascalVOCDataset,BalloonDataset,FruitsNutsDataset
from data.msCOCODatas import MSCOCOKeypointDataset
from data.augment import bboxAug
from tools.visual.vis import vis_rect,vis_keypoints,vis_keypoints2

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
    root = "/media/wucong/225A6D42D4FA828F1/datas/COCO"
    classes = ["person"]
    do_segm = False
    do_mask = False
    do_keypoint = True
    # dataset = MSCOCODataset(root,mode="val", do_segm=do_segm, do_mask=do_mask, do_keypoint=do_keypoint)
    typeOfData = None

    # """

    # """
    # root = r"/media/wucong/225A6D42D4FA828F1/datas/PennFudanPed"
    # classes = ["person"]
    # typeOfData = "PennFudanDataset"


    # root = r"/media/wucong/225A6D42D4FA828F1/datas/balloon_dataset/balloon/train"
    # classes = ["balloon"]
    # typeOfData = "BalloonDataset"

    # root = r"/media/wucong/225A6D42D4FA828F1/datas/data"
    # classes = ["date","fig","hazelnut"]
    # typeOfData = "FruitsNutsDataset"
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

    train_transforms = bboxAug.Compose([
               # bboxAug.RandomChoice(),
               bboxAug.RandomHorizontalFlip(),
               bboxAug.RandomBrightness(),
               bboxAug.RandomBlur(),
               bboxAug.RandomSaturation(),
               bboxAug.RandomHue(),
               # bboxAug.RandomRotate(angle=5),
               # bboxAug.RandomTranslate(),
               # bboxAug.Augment(False),
               bboxAug.Pad(), bboxAug.Resize((416,416), False),
               # bboxAug.ResizeMinMax(800,1333),
               # bboxAug.ResizeFixAndPad(),
               # bboxAug.RandomHSV(),
               # bboxAug.RandomCutout(),
               bboxAug.ToTensor(), # PIL --> tensor
               # bboxAug.Normalize() # tensor --> tensor
           ])

    if typeOfData == "PennFudanDataset":
        Data = PennFudanDataset
    elif typeOfData == "PascalVOCDataset":
        Data = PascalVOCDataset
    elif typeOfData == "BalloonDataset":
        Data = BalloonDataset
    elif typeOfData == "FruitsNutsDataset":
        Data = FruitsNutsDataset
    else:
        Data = None

    # dataset = Data(root, 2012, transforms=train_transforms, classes=classes, useMosaic=True)

    dataset = MSCOCOKeypointDataset(root,mode="minival",transforms=train_transforms,classes=classes,useMosaic=False)
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

            for idx,(box,label) in enumerate(zip(boxes,labels)):
                data = vis_rect(data,box,str(label),0.5,label)

                if "keypoints" in target:
                    data = vis_keypoints2(data, target["keypoints"].to("cpu").numpy().transpose([0,2,1]),1)

            cv2.imshow("test", data)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__=="__main__":
    test_datasets()