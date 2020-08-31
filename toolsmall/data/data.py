import os
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
import torch.utils.data

from toolsmall.data import bboxAug,PennFudanDataset,PascalVOCDataset,ValidDataset,BalloonDataset,FruitsNutsDataset

def collate_fn(batch_data):
    data_list = []
    target_list = []
    for data,target in batch_data:
        data_list.append(data)
        target_list.append(target)

    return data_list,target_list

def get_transform(train=True,min_size=800,max_size=1333,useImgaug=True,advanced=False):
    if train:
        if useImgaug:
            transforms = bboxAug.Compose([
                bboxAug.Augment(advanced=advanced),
                bboxAug.ResizeMinMax(min_size,max_size),
                bboxAug.ToTensor(),  # PIL --> tensor
                # bboxAug.Normalize()  # tensor --> tensor
                bboxAug.Normalize(image_std=[0.229, 0.224, 0.225])
            ])
        else:
            transforms = bboxAug.Compose([
                # bboxAug.RandomChoice(),
                bboxAug.RandomHorizontalFlip(),
                # bboxAug.RandomBrightness(),
                # bboxAug.RandomBlur(),
                # bboxAug.RandomSaturation(),
                # bboxAug.RandomHue(),
                # bboxAug.RandomRotate(angle=5),
                # bboxAug.RandomTranslate(),
                bboxAug.ResizeMinMax(min_size, max_size),
                bboxAug.ToTensor(),  # PIL --> tensor
                # bboxAug.Normalize()  # tensor --> tensor
                bboxAug.Normalize(image_std=[0.229, 0.224, 0.225])
            ])
    else:
        transforms = bboxAug.Compose([
            bboxAug.ResizeMinMax(min_size, max_size),
            bboxAug.ToTensor(),  # PIL --> tensor
            # bboxAug.Normalize()  # tensor --> tensor
            bboxAug.Normalize(image_std=[0.229, 0.224, 0.225])
        ])

    return transforms

class Datas:
    def __init__(self,isTrain=False,trainDP=None,testDP=None,predDP=None,
                 typeOfData="PennFudanDataset",classes=[],
                 num_workers=5,min_size=800,max_size=1333,batch_size=2,
                 useImgaug=True,advanced=False,seed=100):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        torch.manual_seed(seed)
        kwargs = {'num_workers': num_workers, 'pin_memory': True} if self.use_cuda else {}

        train_transforms = get_transform(train=True, min_size=min_size,max_size=max_size, useImgaug=useImgaug, advanced=advanced)
        test_transforms = get_transform(False, min_size=min_size,max_size=max_size)

        if isTrain:
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
            train_dataset = Data(trainDP, 2012, transforms=train_transforms, classes=classes, useMosaic=True)

            if testDP is not None:
                test_dataset = Data(testDP, 2012, transforms=test_transforms, classes=classes)

            else:
                test_dataset = Data(trainDP, 2012, transforms=test_transforms, classes=classes)
                num_datas = len(train_dataset)
                num_train = int(0.8 * num_datas)
                indices = torch.randperm(num_datas).tolist()
                train_dataset = torch.utils.data.Subset(train_dataset, indices[:num_train])
                test_dataset = torch.utils.data.Subset(test_dataset, indices[num_train:])

            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=collate_fn, **kwargs)

            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=collate_fn, **kwargs)

            if predDP is not None:
                pred_dataset = ValidDataset(predDP, transforms=test_transforms)

                self.pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=collate_fn, **kwargs)
            else:
                self.pred_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=collate_fn, **kwargs)

        else:
            pred_dataset = ValidDataset(predDP, transforms=test_transforms)

            self.pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=collate_fn, **kwargs)


if __name__=="__main__":
    # classes = ["__background__","bicycle", "bus", "car", "motorbike", "person"]
    classes = ["__background__", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike",
               "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]
    preddataPath = None
    testdataPath = None
    traindataPath = "/media/wucong/225A6D42D4FA828F1/datas/voc/VOCdevkit/"
    typeOfData = "PascalVOCDataset"

    data=Datas(True,traindataPath,testdataPath,preddataPath,typeOfData,classes,min_size=600,max_size=1000,batch_size=1)