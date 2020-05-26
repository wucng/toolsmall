"""
pytorch自带的一些数据接口，改造以便更方便使用
import torchvision.datasets
"""

import os
import numpy as np
import torch
from PIL import Image
# from torchvision.datasets.voc import VOCDetection,VOCSegmentation
# from torchvision.datasets.coco import CocoCaptions, CocoDetection
from torch.utils.data import Dataset,DataLoader
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.voc import *

__all__=["glob_format","PennFudanDataset","PascalVOCDataset","PascalVOCDataset","ValidDataset"]

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
    def __init__(self, root="./PennFudanPed/", transforms=None,classes=[]):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
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
    def __init__(self, root,year=2007,transforms=None,classes=[],useDifficult=False):
        # self.root = os.path.join(root,"VOCdevkit","VOC%s"%(year))
        self.root = os.path.join(root,"VOC%s"%(year))
        self.transforms = transforms
        self.classes=classes
        self.useDifficult = useDifficult
        self.annotations = self.change2csv()

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
