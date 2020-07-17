"""
pytorch自带的一些数据接口，改造以便更方便使用
import torchvision.datasets

https://zhuanlan.zhihu.com/p/137073821
https://zhuanlan.zhihu.com/p/137387839
"""

import os
import numpy as np
import torch
from PIL import Image
import random
import cv2,json,math
# from torchvision.datasets.voc import VOCDetection,VOCSegmentation
# from torchvision.datasets.coco import CocoCaptions, CocoDetection
from torch.utils.data import Dataset,DataLoader
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.voc import *

from .augment.uils import glob_format,resize_boxes,random_affine,mosaic_resize,mosaic_crop,mosaic_origin,mixup

__all__=["glob_format","PennFudanDataset","PascalVOCDataset","PascalVOCDataset","ValidDataset"]

class PennFudanDataset(object):
    """
    wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip

    PennFudanPed/
      PedMasks/
        FudanPed00001_mask.png
        FudanPed00002_mask.png
        FudanPed00003_mask.png
        FudanPed00004_mask.png
        ...
      PNGImages/
        FudanPed00001.png
        FudanPed00002.png
        FudanPed00003.png
        FudanPed00004.png
    """
    def __init__(self, root="./PennFudanPed/",year=None, transforms=None,classes=[],useMosaic=False):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        self.useMosaic = useMosaic
        self.classes = classes
        assert "person" in self.classes

    def load(self,idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        # img = Image.open(img_path).convert("RGB")
        img = np.asarray(Image.open(img_path).convert("RGB"), np.uint8)
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance ，因为每种颜色对应不同的实例
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)

        # instances are encoded as different colors
        # 实例被编码为不同的颜色（0为背景，1为对象1,2为对象2,3为对象3，...）
        obj_ids = np.unique(mask)  # array([0, 1, 2], dtype=uint8),mask有2个对象分别为1,2
        # first id is the background, so remove it
        # first id是背景，所以删除它
        obj_ids = obj_ids[1:]  # array([1, 2], dtype=uint8)

        # split the color-encoded mask into a set
        # of binary masks ,0,1二值图像
        # 将颜色编码的掩码分成一组二进制掩码 SegmentationObject-->mask
        masks = mask == obj_ids[:, None, None]  # shape (2, 536, 559)，2个mask
        # obj_ids[:, None, None] None为增加对应的维度，shape为 [2, 1, 1]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        labels = []
        for i in range(num_objs):  # mask反算对应的bbox
            pos = np.where(masks[i])  # 返回像素值为1 的索引，pos[0]对应行(y)，pos[1]对应列(x)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.classes.index("person"))


        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # there is only one class
        # labels = torch.zeros((num_objs,), dtype=torch.int64)
        # labels = torch.ones((num_objs,), dtype=torch.int64) # 包括背景，背景默认为0
        # masks = torch.as_tensor(masks, dtype=torch.uint8)

        # return img,masks.transpose([1,2,0]).astype(np.float32), boxes, labels,img_path
        return img,None, boxes, labels,img_path

    def __getitem__(self, idx):
        if self.useMosaic:
            # state = np.random.choice(["general", "ricap", "mixup"], 1)[0]
            state = np.random.choice(["general", "ricap"], 1)[0]
            if state == "general":
                img, masks, boxes, labels,img_path = self.load(idx)
            elif state == "ricap":
                # img, masks, boxes, labels,img_path = mosaic_origin(self,idx)
                # img, masks, boxes, labels,img_path = mosaic_resize(self,idx)
                img, masks, boxes, labels,img_path = mosaic_crop(self,idx)
            else:
                pass
                # img, masks, boxes, labels,img_path = mixup(self,idx)
        else:
            img, masks, boxes, labels,img_path = self.load(idx)

        img = Image.fromarray(img)
        if masks is not None:
            masks = torch.from_numpy(masks)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["path"] = img_path

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class PascalVOCDataset2(VisionDataset):
    """
    http://host.robots.ox.ac.uk/pascal/VOC/
    """
    def __init__(self,
                 root,
                 year='2007',
                 # image_set='train',
                 classes=[],
                 download=False,
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(PascalVOCDataset2, self).__init__(root, transforms, transform, target_transform)
        self.classes = classes
        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        valid_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_sets.append("test")

        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        # image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # self.images = glob_format(image_dir)
        self.annotations = list(sorted(glob_format(annotation_dir)))
        # 过滤掉不在内的数据 classes
        self.annotations = self.filter()
        self.images = [annot.replace("Annotations", "JPEGImages").replace(".xml", ".jpg")
                       for annot in self.annotations]
        assert (len(self.images) == len(self.annotations))


    def filter(self):
        new_annotations = []
        for annot in self.annotations:
            flag = False
            target = self.parse_voc_xml(
                ET.parse(annot).getroot())
            objs = target["annotation"]["object"]
            if not isinstance(objs,list):objs = [objs]
            for obj in objs:
                if obj["name"] in self.classes:
                    flag = True
                else:
                    pass
            if flag:
                new_annotations.append(annot)

        return new_annotations


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        boxes = []
        labels = []
        iscrowd = []
        objs = target["annotation"]["object"]
        if not isinstance(objs, list): objs = [objs]
        for obj in objs:
            name = obj["name"]
            if name not in self.classes:continue
            bndbox = obj["bndbox"]
            bbox = [int(bndbox["xmin"]),int(bndbox["ymin"]),int(bndbox["xmax"]),int(bndbox["ymax"])]
            iscrowd.append(1 if obj["difficult"] else 0)
            labels.append(self.classes.index(name)+1)
            boxes.append(bbox)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        image_id = torch.tensor([index])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["path"] = self.images[index]

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

class PascalVOCDataset(Dataset):
    """
    http://host.robots.ox.ac.uk/pascal/VOC/
    """
    def __init__(self, root,year=2007,transforms=None,classes=[],useDifficult=False,useMosaic=False):
        # self.root = os.path.join(root,"VOCdevkit","VOC%s"%(year))
        self.root = os.path.join(root,"VOC%s"%(year))
        self.transforms = transforms
        self.classes=classes
        self.useDifficult = useDifficult
        self.annotations = self.change2csv()

        self.useMosaic=useMosaic

    def parse_xml(self,xml):
        in_file = open(xml)
        tree = ET.parse(in_file)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.iter('object'):
            if not self.useDifficult:
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

    def load(self,idx):
        annotations = self.annotations[idx]
        img_path = annotations["image"]
        img = np.asarray(Image.open(img_path).convert("RGB"),np.uint8)
        boxes = annotations["boxes"]
        labels = annotations["labels"]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        mask = None
        return img,mask,boxes,labels,img_path

    def __getitem__(self, idx):
        if self.useMosaic:
            # state = np.random.choice(["general", "ricap", "mixup"], 1)[0]
            state = np.random.choice(["general", "ricap"], 1)[0]
            if state == "general":
                img,mask, boxes, labels,img_path = self.load(idx)
            elif state == "ricap":
                # img,mask,boxes, labels,img_path = mosaic_origin(self,idx)
                img,mask,boxes, labels,img_path = mosaic_crop(self,idx)
                # img,mask,boxes, labels,img_path = mosaic_resize(self,idx)
            else:
                # img,mask,boxes, labels,img_path = mixup(self,idx)
                pass
        else:
            img,mask,boxes, labels, img_path = self.load(idx)

        img = Image.fromarray(img)
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

class BalloonDataset(Dataset):
    """
    !wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
    !unzip balloon_dataset.zip > /dev/null
    """
    def __init__(self,root,year=None,transforms=None,classes=[],useMosaic=False):
        # mode="train"
        self.json_file = os.path.join(root,"via_region_data.json")
        with open(self.json_file) as f:
            self.imgs_anns = json.load(f)
        self.keys = list(self.imgs_anns.keys())

        self.transforms = transforms
        self.useMosaic = useMosaic
        self.classes = classes
        assert "balloon" in self.classes

    def __len__(self):
        return len(self.keys)

    def load(self,idx):
        tdata = self.imgs_anns[self.keys[idx]]
        img_path = os.path.join(os.path.dirname(self.json_file),tdata["filename"])
        img = np.asarray(Image.open(img_path).convert("RGB"), np.uint8)

        annos = tdata["regions"]
        boxes = []
        labels = []
        # masks = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            boxes.append([np.min(px), np.min(py), np.max(px), np.max(py)])
            labels.append(self.classes.index("balloon"))
            # masks.append([poly])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # labels = torch.zeros((boxes.size(0),), dtype=torch.int64)
        # labels = torch.ones((boxes.size(0),), dtype=torch.int64) # 包括背景，背景默认为0
        mask = None

        return img,mask,boxes,labels,img_path

    def __getitem__(self, idx):
        if self.useMosaic:
            # state = np.random.choice(["general", "ricap", "mixup"], 1)[0]
            state = np.random.choice(["general", "ricap"], 1)[0]
            if state == "general":
                img,mask,boxes, labels, img_path = self.load(idx)
            elif state == "ricap":
                # img,mask, boxes, labels, img_path = mosaic_origin(self,idx)
                # img,mask, boxes, labels, img_path = mosaic_resize(self,idx)
                img,mask, boxes, labels, img_path = mosaic_crop(self,idx)
            else:
                # img,mask, boxes, labels,img_path = mixup(self,idx)
                pass
        else:
            img,mask, boxes, labels, img_path = self.load(idx)

        img = Image.fromarray(img)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
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

    def __init__(self, root="", year=None, transforms=None, classes=[], useMosaic=False):
        self.root = root
        self.transforms = transforms
        self.classes = classes
        self.useMosaic = useMosaic
        self.json_file = os.path.join(root, "trainval.json")
        with open(self.json_file) as f:
            imgs_anns = json.load(f)

        self.annotations = self.change_csv(imgs_anns)

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
        mask = None
        return img,mask,boxes,labels,img_path

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
            # state = np.random.choice(["general", "ricap", "mixup"], 1)[0]
            state = np.random.choice(["general", "ricap"], 1)[0]
            if state == "general":
                img,mask,boxes, labels, img_path = self.load(idx)
            elif state == "ricap":
                # img,mask, boxes, labels, img_path = mosaic_origin(self,idx)
                # img,mask, boxes, labels, img_path = mosaic_resize(self,idx)
                img,mask, boxes, labels, img_path = mosaic_crop(self,idx)
            else:
                # img,mask, boxes, labels,img_path = mixup(self,idx)
                pass
        else:
            img,mask, boxes, labels, img_path = self.load(idx)

        img = Image.fromarray(img)
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

if __name__=="__main__":
    root = r"C:\practice\data"
    datas = PascalVOCDataset(root,'2007',["person"])
    for img, target in datas:
        print()
