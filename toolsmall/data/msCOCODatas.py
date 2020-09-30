import os,cv2
import torch
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import pickle
from tqdm import tqdm


# only person keypoint
class MSCOCOKeypointDataset2(Dataset):
    """速度很慢,不推荐"""
    def __init__(self, root="", year=2014,mode="val",transforms=None, classes=[], useMosaic=False):
        assert mode in ["train","val","minival","valminusminival"]
        assert year in [2014,2017]
        self.root = root
        self.year = year
        self.mode = mode
        if mode in ["minival","valminusminival"]:mode="val"
        self.imgName = "%s%s"%(mode,year)

        self.transforms = transforms
        self.classes = classes
        self.useMosaic = useMosaic
        self.json_file = os.path.join(root, "annotations","person_keypoints_%s%s.json"%(mode,year))
        with open(self.json_file) as f:
            imgs_anns = json.load(f)

        pklfile = os.path.join(root,"%s.pkl"%self.imgName)
        if not os.path.exists(pklfile):
            annotations = self.change_csv(imgs_anns)
            pickle.dump(annotations,open(pklfile,"wb"))
        else:
            annotations = pickle.load(open(pklfile,"rb"))

        self.annotations = annotations

    def change_csv(self,imgs_anns):
        # 找到每张图片所以标注，组成一个list
        images_dict = {item["id"]: item["file_name"] for item in imgs_anns["images"]}
        categories_dict = {item["id"]: item["name"] for item in imgs_anns["categories"]}
        keypoints_name = imgs_anns["categories"][0]["keypoints"]
        skeleton = imgs_anns["categories"][0]["skeleton"]
        annotations = imgs_anns["annotations"]

        result = []
        for k, v in tqdm(images_dict.items()):
            img_path = os.path.join(self.root,self.imgName,v)
            boxes = []
            labels = []
            iscrowd = []
            # image_id = []
            area = []
            segment = []
            keypoints = []
            for item in annotations:
                if item["image_id"] == k:
                    segment.append(item["segmentation"])
                    iscrowd.append(item["iscrowd"])
                    area.append(item["area"])
                    # boxes.append(item["bbox"])
                    bbox = item["bbox"]
                    # boxes.append([bbox[0]-bbox[2]/2,bbox[1]-bbox[3]/2,bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2])
                    boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                    labels.append(self.classes.index(categories_dict[item["category_id"]]))
                    keypoints.append(item["keypoints"])

                    annotations.remove(item) # 移除该项

            result.append({"img_path": img_path, "segment": segment, "iscrowd": iscrowd, "area": area, "boxes": boxes,
                           "labels": labels,"keypoints":keypoints,"keypoints_name":keypoints_name,"skeleton":skeleton})

        return result

    def load(self,idx):
        annotations = self.annotations[idx]
        img_path = annotations["img_path"]
        img = np.asarray(Image.open(img_path).convert("RGB"), np.uint8)
        boxes = annotations["boxes"]
        labels = annotations["labels"]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        mask = None
        keypoints = annotations["keypoints"]
        keypoints_name = annotations["keypoints_name"]
        skeleton = annotations["skeleton"]
        iscrowd = annotations["iscrowd"]
        area = annotations["area"]

        return img, (iscrowd,area,mask,keypoints,keypoints_name,skeleton),\
               boxes, labels, img_path

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img, (iscrowd, area, mask, keypoints, keypoints_name, skeleton), \
        boxes, labels, img_path = self.load(idx)
        img = Image.fromarray(img)
        iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        area = torch.tensor(area,dtype=torch.float32)
        keypoints = torch.tensor(keypoints,dtype=torch.float32).reshape([-1,17,3])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["path"] = img_path
        target["keypoints"] = keypoints
        target["keypoints_name"] = keypoints_name
        target["skeleton"] = skeleton

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class MSCOCOKeypointDataset(Dataset):
    """
    https://blog.csdn.net/wc781708249/article/details/79603522
    """
    def __init__(self, root="", year=2014,mode="val",transforms=None, classes=[], useMosaic=False):
        assert mode in ["train","val","minival","valminusminival"]
        assert year in [2014,2017]
        self.root = root
        self.year = year
        self.mode = mode
        if mode in ["minival","valminusminival"]:mode="val"
        self.imgName = "%s%s"%(mode,year)

        self.transforms = transforms
        self.classes = classes
        self.useMosaic = useMosaic

        pklfile = os.path.join(root,"%s.pkl"%self.imgName)
        if not os.path.exists(pklfile):
            self.json_file = os.path.join(root, "annotations", "person_keypoints_%s%s.json" % (mode, year))
            with open(self.json_file) as f:
                imgs_anns = json.load(f)
            annotations = self.change_csv(imgs_anns)
            pickle.dump(annotations,open(pklfile,"wb"))
        else:
            annotations = pickle.load(open(pklfile,"rb"))

        self.keypoints_name = annotations.pop("keypoints_name")
        self.skeleton = annotations.pop("skeleton")
        self.keys = list(annotations.keys())
        self.keys.sort()
        self.annotations = annotations

    def change_csv(self,imgs_anns):
        # 找到每张图片所以标注，组成一个list
        images_dict = {item["id"]: item["file_name"] for item in imgs_anns["images"]}
        categories_dict = {item["id"]: item["name"] for item in imgs_anns["categories"]}
        keypoints_name = imgs_anns["categories"][0]["keypoints"]
        skeleton = imgs_anns["categories"][0]["skeleton"]
        annotations = imgs_anns["annotations"]

        result = {"keypoints_name":keypoints_name,"skeleton":skeleton}
        for item in tqdm(annotations):
            image_id = item["image_id"]
            if image_id not in result:
                result[image_id] = {}
            if "img_path" not in result[image_id]:
                img_path = os.path.join(self.root, self.imgName, images_dict[image_id])
                result[image_id]["img_path"] = img_path

            if "segmentation" not in result[image_id]:
                result[image_id]["segmentation"] = []
            result[image_id]["segmentation"].append(item["segmentation"])

            if "iscrowd" not in result[image_id]:
                result[image_id]["iscrowd"] = []
            result[image_id]["iscrowd"].append(item["iscrowd"])

            if "area" not in result[image_id]:
                result[image_id]["area"] = []
            result[image_id]["area"].append(item["area"])

            if "boxes" not in result[image_id]:
                result[image_id]["boxes"] = []
            bbox = item["bbox"]
            result[image_id]["boxes"].append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])


            if "labels" not in result[image_id]:
                result[image_id]["labels"] = []
            result[image_id]["labels"].append(self.classes.index(categories_dict[item["category_id"]]))

            if "keypoints" not in result[image_id]:
                result[image_id]["keypoints"] = []
            result[image_id]["keypoints"].append(item["keypoints"])

        return result

    def load(self,idx):
        annotations = self.annotations[self.keys[idx]]
        img_path = annotations["img_path"]
        img = np.asarray(Image.open(img_path).convert("RGB"), np.uint8)
        boxes = annotations["boxes"]
        labels = annotations["labels"]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        mask = None
        keypoints = annotations["keypoints"]
        # keypoints_name = annotations["keypoints_name"]
        # skeleton = annotations["skeleton"]
        iscrowd = annotations["iscrowd"]
        area = annotations["area"]

        return img, (iscrowd,area,mask,keypoints),\
               boxes, labels, img_path

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img, (iscrowd, area, mask, keypoints), \
        boxes, labels, img_path = self.load(idx)
        img = Image.fromarray(img)
        iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        area = torch.tensor(area,dtype=torch.float32)
        keypoints = torch.tensor(keypoints,dtype=torch.float32).reshape([-1,17,3]) # [x,y,v]
        """
        # 第3个值
        # v=0 表示这个关键点没有标注（这种情况下x=y=v=0）
        # v=1 表示这个关键点标注了但是不可见(被遮挡了）
        # v=2 表示这个关键点标注了同时也可见
        """

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["path"] = img_path
        target["keypoints"] = keypoints
        # target["keypoints_name"] = self.keypoints_name
        # target["skeleton"] = self.skeleton

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


"""
保证每个box 都有 keypoints
"""
class MSCOCOKeypointDatasetV2(Dataset):
    """
    https://blog.csdn.net/wc781708249/article/details/79603522
    """
    def __init__(self, root="", year=2014,mode="val",transforms=None, classes=[], useMosaic=False):
        assert mode in ["train","val","minival","valminusminival"]
        assert year in [2014,2017]
        self.root = root
        self.year = year
        self.mode = mode
        if mode in ["minival","valminusminival"]:mode="val"
        self.imgName = "%s%s"%(mode,year)

        self.transforms = transforms
        self.classes = classes
        self.useMosaic = useMosaic

        pklfile = os.path.join(root,"%s.pkl"%self.imgName)
        if not os.path.exists(pklfile):
            self.json_file = os.path.join(root, "annotations", "person_keypoints_%s%s.json" % (mode, year))
            with open(self.json_file) as f:
                imgs_anns = json.load(f)
            annotations = self.change_csv(imgs_anns)
            pickle.dump(annotations,open(pklfile,"wb"))
        else:
            annotations = pickle.load(open(pklfile,"rb"))

        self.keypoints_name = annotations.pop("keypoints_name")
        self.skeleton = annotations.pop("skeleton")
        self.keys = list(annotations.keys())
        self.keys.sort()
        self.annotations = annotations

    def change_csv(self,imgs_anns):
        # 找到每张图片所以标注，组成一个list
        images_dict = {item["id"]: item["file_name"] for item in imgs_anns["images"]}
        categories_dict = {item["id"]: item["name"] for item in imgs_anns["categories"]}
        keypoints_name = imgs_anns["categories"][0]["keypoints"]
        skeleton = imgs_anns["categories"][0]["skeleton"]
        annotations = imgs_anns["annotations"]

        # result = {"keypoints_name":keypoints_name,"skeleton":skeleton}
        result = {}
        for item in tqdm(annotations):
            image_id = item["image_id"]
            if image_id not in result:
                result[image_id] = {}
            if "img_path" not in result[image_id]:
                img_path = os.path.join(self.root, self.imgName, images_dict[image_id])
                result[image_id]["img_path"] = img_path

            if "keypoints" not in result[image_id]:
                result[image_id]["keypoints"] = []

            if sum(item["keypoints"])==0:continue # 如果box没有keypoints 剔除掉

            result[image_id]["keypoints"].append(item["keypoints"])

            if "segmentation" not in result[image_id]:
                result[image_id]["segmentation"] = []
            result[image_id]["segmentation"].append(item["segmentation"])

            if "iscrowd" not in result[image_id]:
                result[image_id]["iscrowd"] = []
            result[image_id]["iscrowd"].append(item["iscrowd"])

            if "area" not in result[image_id]:
                result[image_id]["area"] = []
            result[image_id]["area"].append(item["area"])

            if "boxes" not in result[image_id]:
                result[image_id]["boxes"] = []
            bbox = item["bbox"]
            result[image_id]["boxes"].append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])


            if "labels" not in result[image_id]:
                result[image_id]["labels"] = []
            result[image_id]["labels"].append(self.classes.index(categories_dict[item["category_id"]]))



        # 如果box没有keypoints 剔除掉
        tmp = {"keypoints_name":keypoints_name,"skeleton":skeleton}
        for k,v in result.items():
            if len(result[k]["keypoints"])>0:
                tmp[k] = v
        result = tmp

        return result

    def load(self,idx):
        annotations = self.annotations[self.keys[idx]]
        img_path = annotations["img_path"]
        img = np.asarray(Image.open(img_path).convert("RGB"), np.uint8)
        boxes = annotations["boxes"]
        labels = annotations["labels"]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        mask = None
        keypoints = annotations["keypoints"]
        # keypoints_name = annotations["keypoints_name"]
        # skeleton = annotations["skeleton"]
        iscrowd = annotations["iscrowd"]
        area = annotations["area"]

        return img, (iscrowd,area,mask,keypoints),\
               boxes, labels, img_path

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img, (iscrowd, area, mask, keypoints), \
        boxes, labels, img_path = self.load(idx)
        img = Image.fromarray(img)
        iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        area = torch.tensor(area,dtype=torch.float32)
        keypoints = torch.tensor(keypoints,dtype=torch.float32).reshape([-1,17,3]) # [x,y,v]
        """
        # 第3个值
        # v=0 表示这个关键点没有标注（这种情况下x=y=v=0）
        # v=1 表示这个关键点标注了但是不可见(被遮挡了）
        # v=2 表示这个关键点标注了同时也可见
        """

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["path"] = img_path
        target["keypoints"] = keypoints
        # target["keypoints_name"] = self.keypoints_name
        # target["skeleton"] = self.skeleton

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

class MSCOCOKeypointDatasetV3(Dataset):
    """
    https://blog.csdn.net/wc781708249/article/details/79603522
    """
    def __init__(self, root="", year=2014,mode="val",transforms=None, classes=[], useMosaic=False):
        assert mode in ["train","val","minival","valminusminival"]
        assert year in [2014,2017]
        self.root = root
        self.year = year
        self.mode = mode
        if mode in ["minival","valminusminival"]:mode="val"
        self.imgName = "%s%s"%(mode,year)

        self.transforms = transforms
        self.classes = classes
        self.useMosaic = useMosaic

        pklfile = os.path.join(root,"%s.pkl"%self.imgName)
        if not os.path.exists(pklfile):
            self.json_file = os.path.join(root, "annotations", "person_keypoints_%s%s.json" % (mode, year))
            with open(self.json_file) as f:
                imgs_anns = json.load(f)
            annotations = self.change_csv(imgs_anns)
            pickle.dump(annotations,open(pklfile,"wb"))
        else:
            annotations = pickle.load(open(pklfile,"rb"))

        self.keypoints_name = annotations.pop("keypoints_name")
        self.skeleton = annotations.pop("skeleton")
        self.keys = list(annotations.keys())
        self.keys.sort()
        self.annotations = annotations

    def change_csv(self,imgs_anns):
        # 找到每张图片所以标注，组成一个list
        images_dict = {item["id"]: item["file_name"] for item in imgs_anns["images"]}
        categories_dict = {item["id"]: item["name"] for item in imgs_anns["categories"]}
        keypoints_name = imgs_anns["categories"][0]["keypoints"]
        skeleton = imgs_anns["categories"][0]["skeleton"]
        annotations = imgs_anns["annotations"]

        # result = {"keypoints_name":keypoints_name,"skeleton":skeleton}
        result = {}
        for item in tqdm(annotations):
            image_id = item["image_id"]
            if image_id not in result:
                result[image_id] = {}
            if "img_path" not in result[image_id]:
                img_path = os.path.join(self.root, self.imgName, images_dict[image_id])
                result[image_id]["img_path"] = img_path

            if "keypoints" not in result[image_id]:
                result[image_id]["keypoints"] = []

            # if sum(item["keypoints"])==0:continue # 如果box没有keypoints 剔除掉
            # if np.sum(np.array(item["keypoints"]).reshape(-1,3)[:,-1]>0) < 8:continue
            # if np.sum(np.array(item["keypoints"]).reshape(-1,3)[:,-1]>0) < 12:continue
            # if np.sum(np.array(item["keypoints"]).reshape(-1,3)[:,-1]>0) < 14:continue
            if np.sum(np.array(item["keypoints"]).reshape(-1,3)[:,-1]>0) >=15:
                result[image_id]["keypoints"].append(item["keypoints"])

            if "segmentation" not in result[image_id]:
                result[image_id]["segmentation"] = []
            result[image_id]["segmentation"].append(item["segmentation"])

            if "iscrowd" not in result[image_id]:
                result[image_id]["iscrowd"] = []
            result[image_id]["iscrowd"].append(item["iscrowd"])

            if "area" not in result[image_id]:
                result[image_id]["area"] = []
            result[image_id]["area"].append(item["area"])

            if "boxes" not in result[image_id]:
                result[image_id]["boxes"] = []
            bbox = item["bbox"]
            result[image_id]["boxes"].append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])


            if "labels" not in result[image_id]:
                result[image_id]["labels"] = []
            result[image_id]["labels"].append(self.classes.index(categories_dict[item["category_id"]]))


        # 如果box没有keypoints 剔除掉
        tmp = {"keypoints_name":keypoints_name,"skeleton":skeleton}
        for k,v in result.items():
            if len(result[k]["keypoints"])==len(result[k]["boxes"]):
                tmp[k] = v
        result = tmp

        return result

    def load(self,idx):
        annotations = self.annotations[self.keys[idx]]
        img_path = annotations["img_path"]
        img = np.asarray(Image.open(img_path).convert("RGB"), np.uint8)
        boxes = annotations["boxes"]
        labels = annotations["labels"]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        mask = None
        keypoints = annotations["keypoints"]
        # keypoints_name = annotations["keypoints_name"]
        # skeleton = annotations["skeleton"]
        iscrowd = annotations["iscrowd"]
        area = annotations["area"]

        return img, (iscrowd,area,mask,keypoints),\
               boxes, labels, img_path

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img, (iscrowd, area, mask, keypoints), \
        boxes, labels, img_path = self.load(idx)
        img = Image.fromarray(img)
        iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        area = torch.tensor(area,dtype=torch.float32)
        keypoints = torch.tensor(keypoints,dtype=torch.float32).reshape([-1,17,3]) # [x,y,v]
        """
        # 第3个值
        # v=0 表示这个关键点没有标注（这种情况下x=y=v=0）
        # v=1 表示这个关键点标注了但是不可见(被遮挡了）
        # v=2 表示这个关键点标注了同时也可见
        """

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["path"] = img_path
        target["keypoints"] = keypoints
        # target["keypoints_name"] = self.keypoints_name
        # target["skeleton"] = self.skeleton

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

if __name__=="__main__":
    classes = ["__background__","person"]
    dataset = MSCOCOKeypointDataset("/media/wucong/225A6D42D4FA828F1/datas/COCO",mode="minival",classes=classes)
    for image,target in dataset:
        pass