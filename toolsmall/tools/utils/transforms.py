import random
import torch
import numpy as np
import cv2
import math
import random
import PIL.Image

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def resize_boxes(boxes, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

def resize_keypoints(keypoints, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_height, ratio_width = ratios

    keypoints[...,0] *= ratio_width  # x
    keypoints[...,1] *= ratio_height # y

    return keypoints

def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # targets = [cls, xyxy]

    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets

def mosaic(self,index): # 会变成固定 h:w =1:1
    # 做马赛克数据增强，详情参考：yolov4
    self.hyp={
        'degrees': 1.98 * 0,  # image rotation (+/- deg)
        'translate': 0.05 * 0,  # image translation (+/- fraction)
        'scale': 0.05 * 0,  # image scale (+/- gain)
        'shear': 0.641 * 0}  # image shear (+/- deg)

    img, _, _, img_path = self.load(index)

    labels4 = []
    s = max(img.shape[:2])
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    indices = [index] + [random.randint(0, self.__len__() - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        img,boxes, labels,_ = self.load(index)

        h, w = img.shape[:2]
        labels = torch.cat((labels.float().unsqueeze(-1),boxes),-1)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114., dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b
        if len(labels)>0:
            labels = labels+torch.tensor([0,padw,padh,padw,padh]).unsqueeze(0)
        labels4.append(labels.numpy())

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

    # Augment
    # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
    img4, labels4 = random_affine(img4, labels4,
                                  degrees=self.hyp['degrees'],
                                  translate=self.hyp['translate'],
                                  scale=self.hyp['scale'],
                                  shear=self.hyp['shear'],
                                  border=-s // 2)  # border to remove

    if len(labels4)>0:
        boxes = torch.tensor(labels4[:,1:]).float()
        labels = torch.as_tensor(labels4[:,0], dtype=torch.long)

        return img4,boxes,labels,img_path
    else:
        img, boxes, labels, img_path = self.load(index)
        return img, boxes, labels, img_path


"""过滤掉很小的框"""
def filter(target,imgSize=(),minhw=20):
    h,w = imgSize
    boxes = target["boxes"]
    # 裁剪到指定范围
    boxes[...,[0,2]] = boxes[...,[0,2]].clamp(0,w-1)
    boxes[...,[1,3]] = boxes[...,[1,3]].clamp(0,h-1)

    wh = boxes[...,2:]-boxes[...,:2]
    keep = (wh > minhw).sum(-1)==2
    boxes = boxes[keep]
    target["boxes"] = boxes
    target["labels"] = target["labels"][keep]
    if len(boxes) == 0:
        return None

    return target

# 4张图片做 mosaic
def mosaicFourImg(self,idx,alpha=0.5):
    try:
        _boxes = []
        _labels = []

        index = torch.randperm(self.__len__()).tolist()
        if idx + 3 >= self.__len__():
            idx = 0

        idx2 = index[idx + 1]
        idx3 = index[idx + 2]
        idx4 = index[idx + 3]

        img,  boxes, labels, img_path = self.load(idx)
        img2,  boxes2, labels2, _ = self.load(idx2)
        img3,  boxes3, labels3, _ = self.load(idx3)
        img4,  boxes4, labels4, _ = self.load(idx4)

        boxes, labels = boxes.numpy(), labels.numpy()
        boxes2, labels2 = boxes2.numpy(), labels2.numpy()
        boxes3, labels3 = boxes3.numpy(), labels3.numpy()
        boxes4, labels4 = boxes4.numpy(), labels4.numpy()


        h1, w1, channel = img.shape
        h2, w2, _ = img2.shape
        h3, w3, _ = img3.shape
        h4, w4, _ = img4.shape

        height = min((h1, h2, h3, h4))
        width = min((w1, w2, w3, w4))
        # height = max((h1, h2, h3, h4))
        # width = max((w1, w2, w3, w4))


        newImg = np.ones((height,width,channel),img.dtype)*114.0
        cy, cx = height // 2, width // 2

        x = random.randint(cx * (1 - alpha), cx * (1 + alpha))
        y = random.randint(cy * (1 - alpha), cy * (1 + alpha))

        # 左上角
        y1 = random.randint(0, h1 - y)
        x1 = random.randint(0, w1 - x)
        newImg[0:y, 0:x] = img[y1:y1 + y, x1:x1 + x]
        tboxes = boxes.copy()
        tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + x) - x1
        tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + y) - y1
        _boxes.extend(tboxes)
        _labels.extend(labels)

        # 右上角
        y1 = random.randint(0, h2 - y)
        x1 = random.randint(0, w2+x-width)
        newImg[0:y, x:width] = img2[y1:y1 + y, x1:x1 + width - x]
        tboxes = boxes2.copy()
        tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + width - x) - x1 + x
        tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + y) - y1

        _boxes.extend(tboxes)
        _labels.extend(labels2)

        # 右下角
        y1 = random.randint(0, h3+y-height)
        x1 = random.randint(0, w3+x-width)
        newImg[y:height, x:width] = img3[y1:y1 + height - y, x1:x1 + width - x]
        tboxes = boxes3.copy()
        tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + width - x) - x1 + x
        tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + height - y) - y1 + y

        _boxes.extend(tboxes)
        _labels.extend(labels3)

        # 左下角
        y1 = random.randint(0, h4+y-height)
        x1 = random.randint(0, w4-x)
        newImg[y:height, 0:x] = img4[y1:y1 + height - y, x1:x1 + x]
        tboxes = boxes4.copy()
        tboxes[..., [0, 2]] = tboxes[..., [0, 2]].clip(x1, x1 + x) - x1
        tboxes[..., [1, 3]] = tboxes[..., [1, 3]].clip(y1, y1 + height - y) - y1 + y

        _boxes.extend(tboxes)
        _labels.extend(labels4)

        target = {}

        target["boxes"] = torch.from_numpy(np.asarray(_boxes)).float()
        target["labels"] = torch.tensor(_labels)

        target = filter(target, (height, width))
        if target is None:
            img,  boxes, labels, img_path = self.load(idx)
        else:
            img = newImg
            boxes = target["boxes"]
            labels = target["labels"]

        return img,  boxes, labels, img_path
    except:
        img,  boxes, labels, img_path = self.load(idx)
        return img,  boxes, labels, img_path


class ResizeMinMax(object):
    """按最小边填充"""
    def __init__(self,min_size=600,max_size=1000): # 800,1333
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img:torch.Tensor,target):
        """
        :param image: torch.Tensor  c,h,w
        :param target: Tensor
        :return:
                image: torch.Tensor
                target: Tensor
        """

        img_h, img_w = img.shape[1:]

        target["original_size"] = torch.as_tensor((img_h, img_w),dtype=torch.float32)

        # 按最小边填充
        min_size = min(img_w, img_h)
        max_size = max(img_w, img_h)
        scale = self.min_size/min_size
        if max_size*scale>self.max_size:
            scale = self.max_size /max_size

        new_w = scale * img_w
        new_h = scale * img_h

        target["resize"] = torch.as_tensor((new_h,new_w,scale), dtype=torch.float32)

        img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale,
                                              recompute_scale_factor=True,
                                              mode="bilinear", align_corners=False)[0]

        if "boxes" in target:
            boxes = target["boxes"]
            boxes = resize_boxes(boxes, (img_h, img_w), (new_h,new_w))
            target["boxes"] = boxes

        if "masks" in target and target["masks"] is not None:
            target["masks"] = torch.nn.functional.interpolate(target["masks"][None].float(),
                                                      size=(new_h,new_w), mode="nearest")[0]#.byte().permute(1,2,0)


        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints,(img_h, img_w), (new_h,new_w))
            target["keypoints"] = keypoints

        return img,target

class ResizeFixSize(object):
    """按最小边填充"""
    def __init__(self,size=416):
        self.size = size

    def __call__(self, img:torch.Tensor,target):
        """
        :param image: torch.Tensor  c,h,w
        :param target: Tensor
        :return:
                image: torch.Tensor
                target: Tensor
        """

        img_h, img_w = img.shape[1:]

        target["original_size"] = torch.as_tensor((img_h, img_w),dtype=torch.float32)

        # 按最小边填充
        max_size = max(img_w, img_h)
        scale = self.size /max_size

        new_w = scale * img_w
        new_h = scale * img_h

        target["resize"] = torch.as_tensor((new_h,new_w,scale), dtype=torch.float32)

        img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale,
                                              recompute_scale_factor=True,
                                              mode="bilinear", align_corners=False)[0] # 已经转换到0~1

        # 填充成 size x size
        tmp = torch.ones([3,self.size,self.size],device=img.device,dtype=img.dtype)*(114./255)
        # 从左上角填充，这样bound box 不变
        tmp[:,:int(new_h),:int(new_w)] = img
        img = tmp

        if "boxes" in target:
            boxes = target["boxes"]
            boxes = resize_boxes(boxes, (img_h, img_w), (new_h,new_w))
            target["boxes"] = boxes

        if "masks" in target and target["masks"] is not None:
            target["masks"] = torch.nn.functional.interpolate(target["masks"][None].float(),
                                                      size=(new_h,new_w), mode="nearest")[0]#.byte().permute(1,2,0)


        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints,(img_h, img_w), (new_h,new_w))
            target["keypoints"] = keypoints

        return img,target

class ResizeMinMax_2(object):
    """按最小边填充"""
    def __init__(self,min_size=600,max_size=1000): # 800,1333
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img,target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image: PIL image
                target: Tensor
        """
        img = np.asarray(img)
        img_h, img_w = img.shape[:2]

        target["original_size"] = torch.as_tensor((img_h, img_w),dtype=torch.float32)

        # 按最小边填充
        min_size = min(img_w, img_h)
        max_size = max(img_w, img_h)
        scale = self.min_size/min_size
        if max_size*scale>self.max_size:
            scale = self.max_size /max_size

        new_w = int(scale * img_w)
        new_h = int(scale * img_h)

        target["resize"] = torch.as_tensor((new_h,new_w,scale), dtype=torch.float32)

        # img = scipy.misc.imresize(img, [new_h,new_w], 'bicubic')  # or 'cubic'
        img = cv2.resize(img,(new_w,new_h),interpolation=cv2.INTER_CUBIC)

        if "boxes" in target:
            boxes = target["boxes"]
            boxes = resize_boxes(boxes, (img_h, img_w), (new_h,new_w))
            target["boxes"] = boxes

        if "masks" in target and target["masks"] is not None:
            target["masks"] = torch.nn.functional.interpolate(target["masks"][None].float(),
                                                      size=(new_h,new_w), mode="nearest")[0].byte()#.permute(1,2,0)


        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints,(img_h, img_w), (new_h,new_w))
            target["keypoints"] = keypoints


        return PIL.Image.fromarray(img),target

class Normalize(object):
    def __init__(self,image_mean=None,image_std=None):
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406] # RGB格式
        if image_std is None:
            image_std = [0.229, 0.224, 0.225] # ImageNet std
            # image_std = [1.0, 1.0, 1.0]
        self.image_mean = image_mean
        self.image_std = image_std

    def __call__(self, image, target):
        """
        :param image: Tensor
        :param target: Tensor
        :return:
                image: Tensor
                target: Tensor
        """
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=torch.float32, device=device)
        std = torch.as_tensor(self.image_std, dtype=torch.float32, device=device)
        image = (image - mean[:, None, None]) / std[:, None, None]
        return image,target

class Pad(object):
    def __init__(self,mode='constant', value=114.):
        self.mode = mode
        self.value = value
    def __call__(self, img,target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image: PIL image
                target: Tensor
        """
        img = np.asarray(img)
        target["original_size"] = torch.as_tensor(img.shape[:2],dtype=torch.float32)
        img,target = self.pad_img(img,target)
        return img,target

    def pad_img(self, img,target):
        h, w = img.shape[:2]
        if "boxes" in target:
            boxes = target["boxes"]
        if h >= w:
            diff = h - w
            pad_list = [[0, 0], [diff // 2, diff - diff // 2], [0, 0]]
            if "boxes" in target:
                boxes = [[b[0] + diff // 2, b[1], b[2] + diff - diff // 2, b[3]] for b in boxes]
                boxes = torch.as_tensor(boxes,dtype=torch.float32)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints[...,0] += diff // 2 # x
                # keypoints[...,1] # y
                target["keypoints"] = keypoints


        else:
            diff = w - h
            pad_list = [[diff // 2, diff - diff // 2], [0, 0], [0, 0]]
            if "boxes" in target:
                boxes = [[b[0], b[1] + diff // 2, b[2], b[3] + diff - diff // 2] for b in boxes]
                boxes = torch.as_tensor(boxes,dtype=torch.float32)

            if "keypoints" in target:
                keypoints = target["keypoints"]
                # keypoints[...,0] = keypoints[...,0]+ diff // 2 # x
                keypoints[...,1] += diff // 2 # y
                target["keypoints"] = keypoints


        img = np.pad(img, pad_list, mode=self.mode, constant_values=self.value)
        if "masks" in target and target["masks"] is not None:
            masks = target["masks"].permute(1,2,0).cpu().numpy() # mask [c,h,w]格式
            masks = np.pad(masks, pad_list, mode=self.mode, constant_values=0)
            target["masks"] = torch.from_numpy(masks).permute(2,0,1)

        if "boxes" in target:
            target["boxes"] = boxes

        return PIL.Image.fromarray(img),target

class Resize(object):
    """按最小边填充"""
    def __init__(self,size=416):
        self.size = size

    def __call__(self, img:torch.Tensor,target):
        """
        :param image: torch.Tensor  c,h,w
        :param target: Tensor
        :return:
                image: torch.Tensor
                target: Tensor
        """

        img_h, img_w = img.shape[1:]

        # 按最小边填充
        max_size = max(img_w, img_h)
        scale = self.size /max_size

        new_w = scale * img_w
        new_h = scale * img_h

        # target["resize"] = torch.as_tensor((new_h,new_w,scale), dtype=torch.float32)
        target["resize"] = torch.as_tensor((new_h,new_w), dtype=torch.float32)

        img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale,
                                              recompute_scale_factor=True,
                                              mode="bilinear", align_corners=False)[0] # 已经转换到0~1

        if "boxes" in target:
            boxes = target["boxes"]
            boxes = resize_boxes(boxes, (img_h, img_w), (new_h,new_w))
            target["boxes"] = boxes

        if "masks" in target and target["masks"] is not None:
            target["masks"] = torch.nn.functional.interpolate(target["masks"][None].float(),
                                                      size=(new_h,new_w), mode="nearest")[0]#.byte().permute(1,2,0)


        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints,(img_h, img_w), (new_h,new_w))
            target["keypoints"] = keypoints

        return img,target


# ---------------------------------------------------------------------------------------------------------------------
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

def run_seq():
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
    # image.
    # if not isinstance(images,list):
    #     images=[images]

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image.
    seq = iaa.Sequential(
        [
            #
            # Apply the following augmenters to most images.
            #
            # iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            # iaa.Flipud(0.2),  # vertically flip 20% of all images

            # crop some of the images by 0-10% of their height/width
            # sometimes(iaa.Crop(percent=(0, 0.1))),
            # sometimes(iaa.Crop(percent=(0, 0.05))),

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                # rotate=(-45, 45),
                rotate=(-5, 5),
                # shear=(-16, 16),
                shear=(-5, 5),
                order=[0, 1],
                # cval=(0, 255),
                cval= 144, # 填充像素值
                # mode=ia.ALL  # 默认常数值填充边界
            )),

            #
            # Execute 0 to 5 of the following (less important) augmenters per
            # image. Don't execute all of them, as that would often be way too
            # strong.
            #
            iaa.SomeOf((0, 5),
                       [
                           # Convert some images into their superpixel representation,
                           # sample between 20 and 200 superpixels per image, but do
                           # not replace all superpixels with their average, only
                           # some of them (p_replace).
                           sometimes(
                               iaa.Superpixels(
                                   p_replace=(0, 1.0),
                                   n_segments=(20, 200)
                               )
                           ),

                           # Blur each image with varying strength using
                           # gaussian blur (sigma between 0 and 3.0),
                           # average/uniform blur (kernel size between 2x2 and 7x7)
                           # median blur (kernel size between 3x3 and 11x11).
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(2, 7)),
                               iaa.MedianBlur(k=(3, 11)),
                           ]),

                           # Sharpen each image, overlay the result with the original
                           # image using an alpha between 0 (no sharpening) and 1
                           # (full sharpening effect).
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                           # Same as sharpen, but for an embossing effect.
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                           # Search in some images either for all edges or for
                           # directed edges. These edges are then marked in a black
                           # and white image and overlayed with the original image
                           # using an alpha of 0 to 0.7.
                           sometimes(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0, 0.7)),
                               iaa.DirectedEdgeDetect(
                                   alpha=(0, 0.7), direction=(0.0, 1.0)
                               ),
                           ])),

                           # Add gaussian noise to some images.
                           # In 50% of these cases, the noise is randomly sampled per
                           # channel and pixel.
                           # In the other 50% of all cases it is sampled once per
                           # pixel (i.e. brightness change).
                           iaa.AdditiveGaussianNoise(
                               loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                           ),

                           # Either drop randomly 1 to 10% of all pixels (i.e. set
                           # them to black) or drop them on an image with 2-5% percent
                           # of the original size, leading to large dropped
                           # rectangles.
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               iaa.CoarseDropout(
                                   (0.03, 0.15), size_percent=(0.02, 0.05),
                                   per_channel=0.2
                               ),
                           ]),

                           # Invert each image's channel with 5% probability.
                           # This sets each pixel value v to 255-v.
                           iaa.Invert(0.05, per_channel=True),  # invert color channels

                           # Add a value of -10 to 10 to each pixel.
                           iaa.Add((-10, 10), per_channel=0.5),

                           # Change brightness of images (50-150% of original value).
                           iaa.Multiply((0.5, 1.5), per_channel=0.5),

                           # Improve or worsen the contrast of images.
                           iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

                           # Convert each image to grayscale and then overlay the
                           # result with the original with random alpha. I.e. remove
                           # colors with varying strengths.
                           iaa.Grayscale(alpha=(0.0, 1.0)),

                           # In some images move pixels locally around (with random
                           # strengths).
                           sometimes(
                               iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                           ),

                           # In some images distort local areas with varying strength.
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                       ],
                       # do all of the above augmentations in random order
                       random_order=True
                       )
        ],
        # do all of the above augmentations in random order
        random_order=True
    )

    # images_aug = seq(images=images)

    return seq

def run_seq2():

    # if not isinstance(images,list):
    #     images=[images]
    # sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        # iaa.Fliplr(0.5),  # horizontal flips
        # iaa.Crop(percent=(0, 0.1)),  # random crops
        # iaa.Crop(percent=(0, 0.05)),
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Sometimes(0.5,
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-5, 5),
            # rotate=(-25, 25),
            shear=(-5, 5),
            order=[0, 1],
            cval=144,  # 填充像素值
        ))

    ], random_order=True)  # apply augmenters in random order

    # images_aug = seq(images=images)

    return seq


def simple_agu(image,target,seed=100,advanced=False,shape=(),last_class_id=True):
    """
    :param image:PIL image
    :param labels: [[x1,y1,x2,y2,class_id],[]]
    :return: image:PIL image
    """
    ia.seed(seed)
    image=np.asarray(image)
    labels=[[*box,label] for box,label in zip(target["boxes"].numpy(),target["labels"].numpy())]

    temp=[BoundingBox(*item[:-1],label=item[-1]) for item in labels]
    bbs=BoundingBoxesOnImage(temp,shape=image.shape)

    seq=run_seq() if advanced else run_seq2()
    # seq=run_seq2()

    # Augment BBs and images.
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()  # 处理图像外的边界框
    if shape:
        image_aug=ia.imresize_single_image(image_aug,shape)
        bbs_aug=bbs_aug.on(image_aug)

    # print coordinates before/after augmentation (see below)
    # use .x1_int, .y_int, ... to get integer coordinates
    # for i in range(len(bbs.bounding_boxes)):
    #         before = bbs.bounding_boxes[i]
    #         after = bbs_aug.bounding_boxes[i]
    #         print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
    #             i,
    #             before.x1, before.y1, before.x2, before.y2,
    #             after.x1, after.y1, after.x2, after.y2)
    #         )

    # image with BBs before/after augmentation (shown below)
    """
    image_before = bbs.draw_on_image(image, size=2)
    image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])

    skimage.io.imshow(image_before)
    skimage.io.show()

    skimage.io.imshow(image_after)
    skimage.io.show()
    # """

    # return image_aug, [[item.x1, item.y1, item.x2, item.y2, item.label] for item in bbs_aug.bounding_boxes] if last_class_id \
    #     else [[item.label,item.x1, item.y1, item.x2, item.y2] for item in bbs_aug.bounding_boxes]

    image_aug=PIL.Image.fromarray(image_aug)

    box=[]
    label=[]
    for item in bbs_aug.bounding_boxes:
        box.append([item.x1, item.y1, item.x2, item.y2])
        label.append(item.label)

    target["boxes"]=torch.as_tensor(box,dtype=torch.float32)
    target["labels"]=torch.as_tensor(label,dtype=torch.long)

    return image_aug,target

class Augment(object):
    def __init__(self,advanced=False):
        self.advanced = advanced
    def __call__(self, image, target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image: PIL image
                target: Tensor
        """
        try:
            _image, _target=simple_agu(image.copy(),target.copy(),np.random.randint(0, int(1e5), 1)[0],self.advanced)
            if len(_target["boxes"])>0:
                # clip to image
                w,h=_image.size # PIL
                _target["boxes"][:,[0,2]] = _target["boxes"][:,[0,2]].clamp(min=0,max=w)
                _target["boxes"][:,[1,3]] = _target["boxes"][:,[1,3]].clamp(min=0,max=h)
                if (_target["boxes"][:,2:]-_target["boxes"][:,:2]>0).all():
                    image, target=_image, _target
            del _image
            del _target
        except:
            pass
        return image,target