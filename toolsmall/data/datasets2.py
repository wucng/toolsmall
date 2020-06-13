# -*- coding:utf-8 -*-

from __future__ import print_function
import xml.etree.ElementTree as ET
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torch
import json
import cv2
import os
from PIL import Image
import sys
import random
from glob import glob
import re

def glob_format(path,base_name = False):
    #print('--------pid:%d start--------------' % (os.getpid()))
    fmt_list = ('.jpg', '.jpeg', '.png',".xml")
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

def load_classes(path):
    classes = []
    with open(path,'r') as fp:
        while True:
            value = fp.readline().strip()
            if value:
                classes.append(value)
            else:
                break
    return classes

class PascalVOCDataset(Dataset):
    """
    http://host.robots.ox.ac.uk/pascal/VOC/
    """
    def __init__(self, root,year=2007,transforms=None,classes=[]):
        # self.root = os.path.join(root,"VOCdevkit","VOC%s"%(year))
        self.root = os.path.join(root,"VOC%s"%(year))
        self.transforms = transforms
        self.classes=classes
        self.annotations = self.change2csv()

    def parse_xml(self,xml):
        in_file = open(xml)
        tree = ET.parse(in_file)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.classes:# or int(difficult)==1:
                continue
            cls_id = self.classes.index(cls)  # 这里不包含背景，如果要包含背景 只需+1, 0默认为背景
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                 int(xmlbox.find('ymax').text))
            boxes.append(b)
            labels.append(cls_id)

        return boxes,labels

    # 将注释文件转成csv格式：xxx/xxx.jpg x1,y1,x2,y2,label,...
    def change2csv(self):
        annotations = []
        xml_files=list(sorted(glob_format(os.path.join(self.root, "Annotations"))))

        for idx,xml in enumerate(xml_files):
            img_path = xml.replace("Annotations", "JPEGImages").replace(".xml", ".jpg")
            boxes,labels=self.parse_xml(xml)
            if len(labels)>0:
                annotations.append({"image":img_path,"boxes":boxes,"labels":labels})

        return annotations

    def __len__(self):
        return len(self.annotations)

    def _shuffle(self,seed=1):
        random.seed(seed)
        random.shuffle(self.annotations)

    def __getitem__(self, idx):
        annotations= self.annotations[idx]
        img_path = annotations["image"]
        img = Image.open(img_path).convert("RGB")
        boxes=annotations["boxes"]
        labels=annotations["labels"]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class FDDBDataset(Dataset):
    """
    http://vis-www.cs.umass.edu/fddb/
    人脸检测数据集，只有一个类别即人脸
    tar zxf FDDB-folds.tgz FDDB-folds
    tar zxf originalPics.tar.gz originalPics

    原数据注释是采用椭圆格式如下
    <major_axis_radius minor_axis_radius angle center_x center_y 1>
    转成矩形格式(x,y,w,h)为:
    [center_x center_y,minor_axis_radius*2,major_axis_radius*2]
    """
    def __init__(self, root,transforms=None,classes=[]):
        self.img_path = os.path.join(root,"originalPics")
        self.annot_path = os.path.join(root,"FDDB-folds")
        self.transforms = transforms
        self.classes=classes
        self.annotations=self.change2csv()

    # 将注释文件转成csv格式：xxx/xxx.jpg x1,y1,x2,y2,label,...
    def change2csv(self):
        annotations=[]
        txts=glob(os.path.join(self.annot_path,"*ellipseList.txt"))
        for txt in txts:
            fp=open(txt)
            datas=fp.readlines()
            tmp={"image":"","boxes":[]}
            for data in datas:
                data=data.strip() # 去掉末尾换行符
                if "img_" in data:
                    if len(tmp["image"])>0:
                        annotations.append(tmp)
                        tmp = {"image": "", "boxes": []}

                    tmp["image"]=os.path.join(self.img_path,data+".jpg")
                elif len(data)<8:
                    continue
                else:
                    tmp_box=[]
                    box=list(map(float,filter(lambda x:len(x)>0,data.split(" "))))
                    tmp_box.extend(box[3:5]) # cx,cy
                    # tmp_box.extend(box[:2]) # w,h
                    tmp_box.extend([2*box[1],2*box[0]]) # w,h
                    # to x1y1x2y2
                    x1=tmp_box[0]-tmp_box[2]/2
                    y1=tmp_box[1]-tmp_box[3]/2
                    x2 = tmp_box[0] + tmp_box[2] / 2
                    y2 = tmp_box[1] + tmp_box[3] / 2
                    tmp_box=[x1,y1,x2,y2]
                    tmp_box.append(box[2]) # 角度
                    tmp["boxes"].append(tmp_box)
            fp.close()

        return annotations


    def __len__(self):
        return len(self.annotations)

    def _shuffle(self,seed=1):
        random.seed(seed)
        random.shuffle(self.annotations)

    def __getitem__(self, idx):
        annotations= self.annotations[idx]
        img_path = annotations["image"]
        img = Image.open(img_path).convert("RGB")
        boxes = []
        labels=[]

        for box in annotations["boxes"]:
            boxes.append(box[:4])
            labels.append(0) # 只有1个类别

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class WIDERFACEDataset(Dataset):
    """
    http://shuoyang1213.me/WIDERFACE/
    人脸检测数据集，只有一个类别即人脸
    unzip wider_face_split.zip -d wider_face_split
    unzip WIDER_train.zip -d WIDER_train

    原数据注释格式如下
    [left, top, width, height, score]
    x1, y1, w, h, 代表人脸框的位置（检测算法一般都要画个框框把人脸圈出来）
    blur：是模糊度，分三档：0，清晰；1：一般般；2：人鬼难分
    express：表达（什么鬼也没弄明白，反正我训这个用不着）
    illumination：曝光，分正常和过曝
    occlusion：遮挡，分三档。0，无遮挡；1，小遮挡；2，大遮挡；
    invalid：（没弄明白）
    pose：（疑似姿态？分典型和非典型姿态）

    """
    def __init__(self, root,transforms=None,classes=[]):
        self.img_path = os.path.join(root,"WIDER_train")
        self.annot_path = os.path.join(root,"wider_face_split")
        self.transforms = transforms
        self.classes=classes
        self.annotations=self.change2csv()

    # 将注释文件转成csv格式：xxx/xxx.jpg x1,y1,x2,y2,label,...
    def change2csv(self):
        annotations=[]
        # txts=glob(os.path.join(self.annot_path,"*ellipseList.txt"))
        # for txt in txts:
        txt=os.path.join(self.annot_path,"wider_face_train_bbx_gt.txt")
        fp=open(txt)
        datas=fp.readlines()
        tmp={"image":"","boxes":[]}
        for data in datas:
            data=data.strip() # 去掉末尾换行符
            if ".jpg" in data:
                if len(tmp["image"])>0 and len(tmp["boxes"])>0:
                    annotations.append(tmp)
                    tmp = {"image": "", "boxes": []}

                if len(tmp["image"])>0 and len(tmp["boxes"])==0:
                    tmp = {"image": "", "boxes": []}

                tmp["image"]=os.path.join(self.img_path,"images",data)
            elif len(data)<8:
                continue
            else:
                tmp_box=[]
                box=list(map(float,filter(lambda x:len(x)>0,data.split(" "))))
                if int(box[4])==2 or int(box[7])==2:continue
                if int(box[2])*int(box[3])<120:continue
                tmp_box.extend(box[:2]) # x1,y1
                tmp_box.extend(box[2:4]) # w,h
                # to x1y1x2y2
                x1=tmp_box[0]
                y1=tmp_box[1]
                x2 = tmp_box[0] + tmp_box[2]
                y2 = tmp_box[1] + tmp_box[3]
                tmp_box=[x1,y1,x2,y2]
                tmp["boxes"].append(tmp_box)
        fp.close()

        return annotations


    def __len__(self):
        return len(self.annotations)

    def _shuffle(self,seed=1):
        random.seed(seed)
        random.shuffle(self.annotations)

    def __getitem__(self, idx):
        annotations= self.annotations[idx]
        img_path = annotations["image"]
        img = Image.open(img_path).convert("RGB")
        boxes = []
        labels=[]

        for box in annotations["boxes"]:
            boxes.append(box[:4])
            labels.append(0) # 只有1个类别

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class PennFudanDataset(Dataset):
    """
    # download the Penn-Fudan dataset
    wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
    # extract it in the current folder
    unzip PennFudanPed.zip
    """
    def __init__(self, root, transforms=None,classes=[]):
        # root=os.path.join(root,"PennFudanPed")
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        # 确保imgs与masks相对应
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance ，因为每种颜色对应不同的实例
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        # 实例被编码为不同的颜色（0为背景，1为对象1,2为对象2,3为对象3，...）
        obj_ids = np.unique(mask) # array([0, 1, 2], dtype=uint8),mask有2个对象分别为1,2
        # first id is the background, so remove it
        # first id是背景，所以删除它
        obj_ids = obj_ids[1:] # array([1, 2], dtype=uint8)

        # split the color-encoded mask into a set
        # of binary masks ,0,1二值图像
        # 将颜色编码的掩码分成一组二进制掩码 SegmentationObject-->mask
        masks = mask == obj_ids[:, None, None] # shape (2, 536, 559)，2个mask
        # obj_ids[:, None, None] None为增加对应的维度，shape为 [2, 1, 1]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs): # mask反算对应的bbox
            pos = np.where(masks[i]) # 返回像素值为1 的索引，pos[0]对应行(y)，pos[1]对应列(x)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.zeros((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = transforms.Compose([
                # transforms.CenterCrop((112,112)),
                transforms.ToTensor()
            ])(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ValidDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.paths = glob_format(root)
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        target = {"path":self.paths[idx]}
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target