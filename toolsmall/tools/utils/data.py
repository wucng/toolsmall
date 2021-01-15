import xml.etree.ElementTree as ET
import numpy as np
import torch
from PIL import Image
import random
import json
from torch.utils.data import Dataset,DataLoader
import pickle
from tqdm import tqdm
import cv2
import os

# from torchvision.datasets.voc import VOCDetection,VOCSegmentation
# from torchvision.datasets.coco import CocoCaptions, CocoDetection
# from torchvision.datasets.vision import VisionDataset
# from torchvision.datasets.voc import *

try:
    from .tools import glob_format,batch,collate_fn
    from .transforms import Compose,RandomHorizontalFlip,ToTensor, \
        mosaicFourImg,mosaic,ResizeFixSize,ResizeMinMax,Normalize,Pad,Resize,Augment
    from .vis import vis_rect
except:
    from tools import glob_format, batch, collate_fn
    from transforms import Compose, RandomHorizontalFlip, ToTensor, \
        mosaicFourImg, mosaic, ResizeFixSize, ResizeMinMax,Normalize,Pad,Resize,Augment
    from vis import vis_rect

class PascalVOCDataset(Dataset):
    """
    http://host.robots.ox.ac.uk/pascal/VOC/
    """
    def __init__(self, root,year=2007,transforms=None,classes=[],useDifficult=False,useMosaic=False,fixsize=False):
        # self.root = os.path.join(root,"VOCdevkit","VOC%s"%(year))
        self.root = os.path.join(root,"VOC%s"%(year))
        self.transforms = transforms
        self.classes=classes
        self.useDifficult = useDifficult

        if not os.path.exists(self.root+".pkl"):
            annotations = self.change2csv()
            pickle.dump(annotations, open(self.root+".pkl", "wb"))
        else:
            annotations = pickle.load(open(self.root+".pkl", "rb"))

        self.annotations = annotations

        self.useMosaic=useMosaic

        self.fixsize = fixsize

    def parse_xml(self,xml):
        in_file = open(xml)
        tree = ET.parse(in_file)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.iter('object'):
            if self.useDifficult:
                pass
            else:
                difficult = obj.find('difficult').text
                if int(difficult) == 1:continue

            cls = obj.find('name').text
            if cls not in self.classes:# or int(difficult)==1:
                continue
            cls_id = self.classes.index(cls)  # 这里不包含背景，如果要包含背景 只需+1, 0默认为背景
            xmlbox = obj.find('bndbox')
            b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)),
                 int(float(xmlbox.find('ymax').text)))
            boxes.append(b)
            labels.append(cls_id)

        return boxes,labels

    # 将注释文件转成csv格式：xxx/xxx.jpg x1,y1,x2,y2,label,...
    def change2csv(self):
        annotations = []
        xml_files=list(sorted(glob_format(os.path.join(self.root, "Annotations"))))

        for idx,xml in tqdm(enumerate(xml_files)):
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

    def load(self,idx):
        annotations = self.annotations[idx]
        img_path = annotations["image"]
        img = np.asarray(Image.open(img_path).convert("RGB"),np.uint8)
        boxes = annotations["boxes"]
        labels = annotations["labels"]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # mask = None
        return img,boxes,labels,img_path

    def get_height_and_width(self,idx):
        # 多GPU训练时有用
        img = self.load(idx)[0]
        return img.shape[:2]

    def __getitem__(self, idx):
        if self.useMosaic:
            if random.random() < 0.5:
                if self.fixsize:
                    img, boxes, labels, img_path = mosaic(self, idx)  # 针对大小固定 对应 yolo，ssd
                else:
                    img,boxes, labels, img_path = mosaicFourImg(self,idx) # 对应 faster rcnn
            else:
                img, boxes, labels, img_path = self.load(idx)
        else:
            img, boxes, labels, img_path = self.load(idx)

        img = Image.fromarray(img.astype(np.uint8))
        iscrowd = torch.zeros_like(labels,dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["path"] = img_path

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

class FruitsNutsDataset(Dataset):
    """
    # download, decompress the data
    !wget https://github.com/Tony607/detectron2_instance_segmentation_demo/releases/download/V0.1/data.zip
    !unzip data.zip > /dev/null
    """

    def __init__(self, root="", year=None, transforms=None, classes=[], useMosaic=False,fixsize=False):
        self.root = root
        self.transforms = transforms
        self.classes = classes
        self.useMosaic = useMosaic
        self.json_file = os.path.join(root, "trainval.json")
        with open(self.json_file) as f:
            imgs_anns = json.load(f)

        self.annotations = self.change_csv(imgs_anns)

        self.fixsize = fixsize

    def __len__(self):
        return len(self.annotations)

    def load(self,idx):
        annotations = self.annotations[idx]
        img_path = annotations["img_path"]
        img = np.asarray(Image.open(img_path).convert("RGB"), np.uint8)
        boxes = annotations["boxes"]
        labels = annotations["labels"]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return img,boxes,labels,img_path

    def get_height_and_width(self,idx):
        # 多GPU训练时有用
        img = self.load(idx)[0]
        return img.shape[:2]

    def change_csv(self,imgs_anns):
        images_dict = {item["id"]: item["file_name"] for item in imgs_anns["images"]}
        categories_dict = {item["id"]: item["name"] for item in imgs_anns["categories"]}
        annotations = imgs_anns["annotations"]

        # 找到每张图片所以标注，组成一个list
        result=[]
        for k,v in images_dict.items():
            img_path = os.path.join(self.root,"images",v)
            boxes = []
            labels = []
            iscrowd = []
            # image_id = []
            area = []
            segment = []
            for item in annotations:
                if item["image_id"]==k:
                    segment.append(item["segmentation"])
                    iscrowd.append(item["iscrowd"])
                    area.append(item["area"])
                    # boxes.append(item["bbox"])
                    bbox = item["bbox"]
                    # boxes.append([bbox[0]-bbox[2]/2,bbox[1]-bbox[3]/2,bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2])
                    boxes.append([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])
                    labels.append(self.classes.index(categories_dict[item["category_id"]]))

            result.append({"img_path":img_path,"segment":segment,"iscrowd":iscrowd,"area":area,"boxes":boxes,
                           "labels":labels})

        return result

    def __getitem__(self, idx):
        if self.useMosaic:
            if random.random() < 0.5:
                if self.fixsize:
                    img, boxes, labels, img_path = mosaic(self, idx)  # 针对大小固定 对应 yolo，ssd
                else:
                    img, boxes, labels, img_path = mosaicFourImg(self, idx)  # 对应 faster rcnn
            else:
                img, boxes, labels, img_path = self.load(idx)
        else:
            img, boxes, labels, img_path = self.load(idx)

        img = Image.fromarray(img.astype(np.uint8))
        iscrowd = torch.zeros_like(labels, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["path"] = img_path

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

def test_data(data_path,img_size,classes,useMosaic,fixsize):
    seed = 100
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(seed)
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}

    transforms = Compose(
        [
            # Augment(False),
            # Pad(),
            # ToTensor(),
            # Resize(img_size),
            # RandomHorizontalFlip(0.5),

            Augment(False),
            ToTensor(),
            ResizeFixSize(img_size) if fixsize else ResizeMinMax(800, 1333),
            RandomHorizontalFlip(0.5),
        ])

    dataset = FruitsNutsDataset(data_path,transforms=transforms,classes=classes,useMosaic=useMosaic,fixsize=fixsize)
    # dataset = PascalVOCDataset(data_path,transforms=transforms,classes=classes,useMosaic=useMosaic,fixsize=fixsize)

    # for img, target in dataset:
    #     print()

    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, **kwargs)

    for datas, targets in data_loader:
        # datas = batch(datas)
        for data, target in zip(datas, targets):
            # from c,h,w ->h,w,c
            data = data.permute(1, 2, 0)
            # to uint8
            data = torch.clamp(data * 255, 0, 255).to("cpu").numpy().astype(np.uint8)

            # to BGR
            # data = data[...,::-1]
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

            boxes = target["boxes"].to("cpu").numpy().astype(np.int)
            labels = target["labels"].to("cpu").numpy()

            for idx, (box, label) in enumerate(zip(boxes, labels)):
                data = vis_rect(data, box, str(label), 0.5, label, useMask=False)

            cv2.imshow("test", data)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# fasterrcnn,rfcn
def get_transform(train=True,fixsize=False,img_size=416,min_size=800,max_size=1333,
                  image_mean=None,image_std=None,advanced=False):
    if image_mean is None:image_mean = [0.485, 0.456, 0.406]
    if image_std is None:image_std = [0.229, 0.224, 0.225]
    if train:
        transforms = Compose(
            [
                Augment(advanced),
                ToTensor(),
                ResizeFixSize(img_size) if fixsize else ResizeMinMax(min_size, max_size),
                RandomHorizontalFlip(0.5),
                Normalize(image_mean,image_std)
            ])
    else:
        transforms = Compose(
            [
                ToTensor(),
                ResizeFixSize(img_size) if fixsize else ResizeMinMax(min_size, max_size),
                # RandomHorizontalFlip(0.5),
                Normalize(image_mean, image_std)
            ])
    return transforms


# ssd ,yolo
def get_transform_fixsize(train=True,img_size=416,
                  image_mean=None,image_std=None,advanced=False):
    if image_mean is None:image_mean = [0.485, 0.456, 0.406]
    if image_std is None:image_std = [0.229, 0.224, 0.225]
    if train:
        transforms = Compose(
            [
                Augment(advanced),
                Pad(),
                ToTensor(),
                Resize(img_size),
                RandomHorizontalFlip(0.5),
                Normalize(image_mean,image_std)
            ])
    else:
        transforms = Compose(
            [
                Pad(),
                ToTensor(),
                Resize(img_size),
                # RandomHorizontalFlip(0.5),
                Normalize(image_mean, image_std)
            ])
    return transforms


class Datas:
    def __init__(self,data_path,year=2012,classes=[],useMosaic=False,fixsize=False,
                 train_ratio = 0.9,batch_size=1,typeOfData = "FruitsNutsDataset",
                 img_size=416,min_size=800,max_size=1333):
        seed = 100
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        torch.manual_seed(seed)
        kwargs = {'num_workers': 5, 'pin_memory': True} if self.use_cuda else {}

        train_transforms = get_transform(True,fixsize,img_size,min_size,max_size)
        test_transforms = get_transform(False,fixsize,img_size,min_size,max_size)

        if typeOfData == "FruitsNutsDataset":
            Data = FruitsNutsDataset
        else:
            Data = PascalVOCDataset

        train_dataset = Data(data_path,year,transforms=train_transforms,classes=classes,useMosaic=useMosaic,fixsize=fixsize)

        test_dataset = Data(data_path, year, transforms=test_transforms, classes=classes,useMosaic=False,fixsize=fixsize)
        num_datas = len(train_dataset)
        num_train = int(train_ratio * num_datas)
        indices = torch.randperm(num_datas).tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, indices[:num_train])
        test_dataset = torch.utils.data.Subset(test_dataset, indices[num_train:])

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                       collate_fn=collate_fn, **kwargs)

        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                      collate_fn=collate_fn, **kwargs)

        self.pred_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                      collate_fn=collate_fn, **kwargs)

if __name__=="__main__":
    data_path = "/media/wucong/225A6D42D4FA828F1/datas/data"
    classes = ["__background__", "date", "fig", "hazelnut"]
    typeOfData = "FruitsNutsDataset"
    test_data(data_path,416,classes,True,False)
    # Datas(data_path,classes=classes,useMosaic=True,fixsize=True,typeOfData=typeOfData)