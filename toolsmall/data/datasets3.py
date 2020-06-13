# -*- coding:utf-8 -*-
"""
进一步工作：结合imgaug做数据增强
https://imgaug.readthedocs.io/en/latest/
"""
from __future__ import print_function
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import xml.etree.ElementTree as ET
import skimage.io as io
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import PIL.ImageDraw
import numpy as np
import torch.utils.data
import torch
import json
import cv2
import os

# import sys
# sys.path.append("../../vision/references/detection")
# import transforms as T

def glob_format(path,base_name = False):
    print('--------pid:%d start--------------' % (os.getpid()))
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
    print('--------pid:%d end--------------' % (os.getpid()))
    return fs

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def convert_coco_poly_to_mask(polygons, height, width):
    rles = coco_mask.frPyObjects(polygons, height, width)
    mask = coco_mask.decode(rles)
    if len(mask.shape) < 3:
        mask = mask[..., None]
    # # mask = torch.as_tensor(mask, dtype=torch.uint8)
    # # mask = mask.any(dim=2)
    mask=np.squeeze(mask,-1)
    return mask

# 从边界点获得mask（边界多边形）
def polygons_to_mask(img_shape, polygons):
    '''
    边界点生成mask
    :param img_shape: [h,w]
    :param polygons: labelme JSON中的边界点格式 [[x1,y1],[x2,y2],[x3,y3],...[xn,yn]]
    :return:
    '''
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=np.uint8) # bool
    return mask


class PennFudanDataset(torch.utils.data.Dataset):
    """
    # download the Penn-Fudan dataset
    wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
    # extract it in the current folder
    unzip PennFudanPed.zip
    """
    def __init__(self, root, transforms=None):
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
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class PascalVOCDataset(torch.utils.data.Dataset):
    """
    http://host.robots.ox.ac.uk/pascal/VOC/
    """
    def __init__(self, root,year=2007,do_segm=False,do_mask=False,transforms=None,classes=[]):
        assert (do_segm and do_mask)==False,"do_segm and do_mask cannot be True at the same time!"
        self.root = os.path.join(root,"VOC%s"%(year))
        self.transforms = transforms
        self.do_segm=do_segm # 是否做实例分割  ([h,w]素取值[0,num_classes](包括背景),0表示背景，1表示类别1,2表示类别2，... )
        self.do_mask=do_mask # 是否做mask  ([N,h,w] # N表示mask个数，每个mask大小hxw与输入的原始图一样大小，每个mask取值为0,1二值图像，0为背景，1为目标)
        self.classes=classes # 只取部分类别
        # load all image files, sorting them to
        # ensure that they are aligned
        # 确保imgs与masks相对应
        if self.do_mask:
            self.segmentationClass = list(sorted(glob_format(os.path.join(self.root, "SegmentationClass"))))
            self.segmentationObject = list(sorted(glob_format(os.path.join(self.root, "SegmentationObject"))))
            # 应该过滤掉没有一个在classes类别的图片
            self.classes_name=["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                    "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        if self.do_segm:
            self.segmentationClass = list(sorted(glob_format(os.path.join(self.root, "SegmentationClass"))))
            # 应该过滤掉没有一个在classes类别的图片
            self.classes_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                                 "diningtable",
                                 "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                                 "tvmonitor"]

        if self.do_mask==False and self.do_segm==False: # 只做bboxs检测
            self.annotations = list(sorted(glob_format(os.path.join(self.root, "Annotations"))))


    def __len__(self):
        return len(self.segmentationClass) if self.do_mask or self.do_segm else len(self.annotations)

    def __getitem__(self, idx):
        if self.do_mask:
            img_path = self.segmentationClass[idx].replace("SegmentationClass", "JPEGImages").replace(".png", ".jpg")
            img = Image.open(img_path).convert("RGB")

            mask = Image.open(self.segmentationObject[idx])
            mask = np.array(mask)
            obj_ids = np.unique(mask)
            # 删除背景 0 与255 (第一个值为0，最后一个为255)
            obj_ids = obj_ids[1:] if 255 not in obj_ids else obj_ids[1:-1]
            # split the color-encoded mask into a set
            # of binary masks ,0,1二值图像
            # 将颜色编码的掩码分成一组二进制掩码 SegmentationObject-->mask
            masks = mask == obj_ids[:, None, None]
            masks = np.asarray(masks, np.uint8)
            # get bounding box coordinates for each mask
            num_objs = len(obj_ids)
            boxes = []
            labels=[]

            mask_cls = Image.open(self.segmentationClass[idx]) # 通过这个确定每个对象的标签
            mask_cls=np.asarray(mask_cls,np.uint8)

            for i in range(num_objs):  # mask反算对应的bbox
                pos = np.where(masks[i])  # 返回像素值为1 的索引，pos[0]对应行(y)，pos[1]对应列(x)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

                mask=mask_cls[ymin:ymax,xmin:xmax]
                _obj_ids = np.unique(mask)
                # 删除背景 0 与255 (第一个值为0，最后一个为255)
                _obj_ids = _obj_ids[1:] if 255 not in _obj_ids else _obj_ids[1:-1]

                max_value=0
                temp=0
                for i in _obj_ids:
                    temp_value=np.sum(mask==i)
                    if temp_value>max_value:
                        max_value=temp_value
                        temp=i

                labels.append(temp)

            boxes=np.asarray(boxes)
            labels=np.asarray(labels)

            # 按classes过滤掉不在这个里面的类别
            temp = []
            for idx,ob in enumerate(labels):
                if self.classes_name[ob - 1] in self.classes:
                    temp.append(idx)

            masks = torch.as_tensor(masks, dtype=torch.uint8)[temp]
            boxes = torch.as_tensor(boxes, dtype=torch.float32)[temp]
            labels = torch.as_tensor(labels, dtype=torch.int64)[temp]

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
            target["masks"] = masks

        if self.do_segm:
            img_path = self.segmentationClass[idx].replace("SegmentationClass", "JPEGImages").replace(".png", ".jpg")
            img = Image.open(img_path).convert("RGB")
            mask=Image.open(self.segmentationClass[idx])
            mask = np.array(mask)
            obj_ids = np.unique(mask)
            # if 255 in obj_ids: # 需删除掉255,0对应背景保留
            #     mask=mask*(mask != 255)

            # 过滤类别
            temp=[]
            for ob in obj_ids:
                if ob ==0 :continue # 背景跳过
                if ob==255:
                    temp.append(ob)
                    continue
                if self.classes_name[ob-1] not in self.classes:
                    temp.append(ob)

            for id in temp:
                mask = mask * (mask != id)

            target=Image.fromarray(mask)

        if self.do_mask==False and self.do_segm==False:
            annotations= self.annotations[idx]
            img_path = self.annotations[idx].replace("Annotations", "JPEGImages").replace(".xml", ".jpg")
            img = Image.open(img_path).convert("RGB")

            in_file=open(annotations)
            tree = ET.parse(in_file)
            root = tree.getroot()
            # size = root.find('size')
            # w = int(size.find('width').text)
            # h = int(size.find('height').text)
            # segmented=int(root.find('segmented').text) # 0不用于分割，1用于分割

            boxes = []
            labels=[]
            for obj in root.iter('object'):
                # difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in self.classes:# or int(difficult)==1:
                    continue
                cls_id = self.classes.index(cls)+1 # 0 默认为背景
                xmlbox = obj.find('bndbox')
                b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),int(xmlbox.find('xmax').text),
                     int(xmlbox.find('ymax').text))
                boxes.append(b)
                labels.append(cls_id)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            # target["image_id"] = image_id
            # target["area"] = area
            # target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

# 需要过滤掉没有标注的图片，否则后面加载数据训练会报错
# 可以根据image_id过滤，即图片id有，但标注中没有该图片对应的image_id需过滤掉
class MSCOCODataset(torch.utils.data.Dataset):
    """
    数据下载:
    https://blog.csdn.net/u014734886/article/details/78830713
    # COCO 格式解析
    https://blog.csdn.net/wc781708249/article/details/79603522
    """
    def __init__(self,root,year=2014,mode="val",do_segm=False,do_mask=False,
                 do_keypoint=False,do_captions=False,transforms=None):

        assert mode in ["train","val"],"mode must be train or val"
        assert year in [2014,2017],"year must be 2014 or 2017"
        assert (do_segm and do_captions)==False,"do_segm and do_captions cannot be True at the same time!"

        self.root=os.path.join(root,"annotations")
        # instances_val2014.json
        # instances_minival2014.json
        # instances_train2014.json
        # instances_valminusminival2014.json
        #
        # person_keypoints_minival2014.json
        # person_keypoints_train2014.json
        # person_keypoints_val2014.json
        # person_keypoints_valminusminival2014.json

        self.year=year
        self.mode=mode
        self.transforms = transforms
        self.do_segm = do_segm  # 是否做实例分割  ([h,w]素取值[0,num_classes](包括背景),0表示背景，1表示类别1,2表示类别2，... )
        self.do_mask = do_mask  # 是否做mask  ([N,h,w] # N表示mask个数，每个mask大小hxw与输入的原始图一样大小，每个mask取值为0,1二值图像，0为背景，1为目标)
        self.do_keypoint = do_keypoint
        self.do_captions=do_captions

        if self.do_captions:
            self.file = os.path.join(self.root, "captions_%s%d.json" % (mode, year))
            data = json.load(open(self.file, 'r'))
        elif self.do_keypoint:
            self.keypoint_file=os.path.join(self.root,"person_keypoints_%s%d.json"%(mode,year))
            data = json.load(open(self.keypoint_file, 'r'))
        else:
            self.file=os.path.join(self.root,"instances_%s%d.json"%(mode,year))
            data = json.load(open(self.file, 'r'))

        self.images=data["images"]
        self.annotations=data["annotations"]

        # 按id从小到大排序
        # self.images=list(sorted(images,key=lambda x:int(x["id"])))
        # self.annotations=list(sorted(annotations,key=lambda x:int(x["image_id"])))

        # 根据image_id过滤，即图片id有，但标注中没有该图片对应的image_id需过滤掉
        # self.image_id=[]
        # for temp_img in self.images:
        #     image_id=temp_img["id"]
        #     if image_id not in self.image_id:
        #         self.image_id.append(image_id)

        # self.image_id = []
        # for ann in self.annotations:
        #     image_id = ann['image_id']
        #     if image_id not in self.image_id:
        #         self.image_id.append(image_id)

        # 直接取标注里的image id
        self.image_id=list(sorted(set([ann['image_id'] for ann in self.annotations])))

        if not self.do_captions:
            # 解析出所有类别
            # 一共80个类别，不包括背景，但其类别id不是连续的且有大于80的id，需按顺序排序后整合成1,2,3...80这种标签
            categories = data["categories"]
            self.classes=[] # 子类别 按id的值顺序插入到list
            self.dict_classes={}
            self.supercategory={} # 父类别
            for cate in categories:
                supercategory=cate["supercategory"]
                id=cate["id"] # 1,2... # 0默认为背景
                name=cate["name"]

                if supercategory not in self.supercategory:
                    self.supercategory[supercategory]=[]
                if name not in self.supercategory[supercategory]:
                    self.supercategory[supercategory].append(name)

                if name not in self.classes:
                    self.classes.append({"id":id,"name":name})

                if name not in self.dict_classes:
                    self.dict_classes.update({id:name})

            # 按id顺序排序，取出正确的次序的label
            self.classes=list(sorted(self.classes,key=lambda x:int(x["id"])))

            self.classes=[item["name"] for item in self.classes]


    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, idx):
        for _image in self.images:
            if _image["id"]==self.image_id[idx]:
                temp_img=_image
                break

        # temp_img=self.images[idx]
        # file_name="COCO_%s%d_%s"%(self.mode,self.year,temp_img["file_name"])
        file_name=temp_img["file_name"]
        image_id=temp_img["id"]

        height=temp_img["height"]
        width=temp_img["width"]


        img_path=os.path.join(self.root.replace("annotations","%s%d"%(self.mode,self.year)),file_name)
        img = Image.open(img_path).convert("RGB")

        if not self.do_captions:
            # 通过imgID 找到其所有对象
            annotation = []
            for ann in self.annotations:
                if ann['image_id'] == image_id:
                    annotation.append(ann)

            labels=[]
            boxes=[]
            # temp_boxes=[]
            # area=[]
            image_id = torch.tensor([idx])
            iscrowd=[]
            masks=[]
            keypoints=[]
            # segmentationClass=[]

            segmentationClass=np.zeros((height,width),np.uint8)

            for ann in annotation:
                try:
                    segment=ann["segmentation"]
                    mask=polygons_to_mask((height,width),np.asarray(segment[0]).reshape([-1,2]))
                    # mask=convert_coco_poly_to_mask(segment,height,width)
                    masks.append(mask)
                    segmentationClass+=mask*ann["category_id"]

                    # mask反算对应的bbox
                    """
                    pos = np.where(mask)  # 返回像素值为1 的索引，pos[0]对应行(y)，pos[1]对应列(x)
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    boxes.append([xmin, ymin, xmax, ymax])
                    """
                    x,y,w,h=ann["bbox"]
                    boxes.append([x,y,x+w,y+h])
                    # """
                    # area.append(ann["area"])
                    iscrowd.append(ann["iscrowd"])
                    labels.append(self.classes.index(self.dict_classes[ann["category_id"]]) + 1)  # 0默认为背景所以从1开始

                    if self.do_keypoint:
                        keypoints.append(np.asarray(ann["keypoints"]).reshape([-1,3]))

                except Exception as e:
                    pass
                    # print(e)
                    # masks.append(np.zeros([height,width],np.uint8))
                    # boxes.append(ann["bbox"])
                    # iscrowd.append(0)
                    # labels.append(0)

        else:
            caption=[]
            # 通过imgID 找到其所有对象
            for ann in self.annotations:
                if ann['image_id'] == image_id:
                    caption.append(ann["caption"])

        if self.do_captions:
            target=caption

        elif self.do_segm:
            target = Image.fromarray(segmentationClass)

        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            if len(boxes)==0:return (img,{})
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # area = torch.as_tensor(area, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            if self.do_mask:
                target["masks"] = masks

            if self.do_keypoint:
                target["masks"] = masks
                target["keypoints"]=torch.as_tensor(keypoints, dtype=torch.float32)


        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

if __name__=="__main__":
    from draw import draw_rect,draw_mask,draw_segms,draw_keypoint
    """
    root = "/media/wucong/d4590a73-a3d9-4971-96fb-4c3cf05abc56/data/PennFudanPed"
    dataset = PennFudanDataset(root, get_transform(train=True))
    """
    # """
    do_segm = True
    do_mask=False
    root = "/media/wucong/d4590a73-a3d9-4971-96fb-4c3cf05abc56/data/VOCdevkit"
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                    "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    dataset = PascalVOCDataset(root, do_segm=do_segm,do_mask=do_mask,transforms=None,classes=classes)
    # """
    """
    root = "/media/wucong/d4590a73-a3d9-4971-96fb-4c3cf05abc56/data/COCO"
    do_segm=False
    do_mask=False
    do_keypoint=True
    dataset=MSCOCODataset(root,do_segm=do_segm,do_mask=do_mask,do_keypoint=do_keypoint)
    """

    for image,target in dataset:
        if not do_segm:
            for k,v in target.items():
                target[k]=v.numpy()
        # draw_rect(image,target)
        # draw_mask(image,target)
        draw_segms(image,target)
        # draw_keypoint(image,target,True,True)
