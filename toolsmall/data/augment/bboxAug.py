"""
图片与bbox一起做数据增强，
如果不传入bbox 就是普通的只做图像增强
"""

import cv2
import numpy as np
import scipy.misc
import skimage
import torch
import PIL.Image
from PIL import Image
import random

from torchvision.transforms import functional as F
from torch.nn import functional as F2
try:
    from .uils import *
except:
    from uils import *

# ----------------------------------------------------------------------------------
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
                image, target=_image, _target
            del _image
            del _target
        except:
            pass
        return image,target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image:Tensor
                target: Tensor
        """
        image = F.to_tensor(image) # 0~1
        return image, target

class Normalize(object):
    def __init__(self,image_mean=None,image_std=None):
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406] # RGB格式
        if image_std is None:
            # image_std = [0.229, 0.224, 0.225] # ImageNet std
            image_std = [1.0, 1.0, 1.0]
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
    def __init__(self,mode='constant', value=128.):
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
        target["original_size"] = torch.as_tensor(img.shape[:2])
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

        else:
            diff = w - h
            pad_list = [[diff // 2, diff - diff // 2], [0, 0], [0, 0]]
            if "boxes" in target:
                boxes = [[b[0], b[1] + diff // 2, b[2], b[3] + diff - diff // 2] for b in boxes]
                boxes = torch.as_tensor(boxes,dtype=torch.float32)

        img = np.pad(img, pad_list, mode=self.mode, constant_values=self.value)

        if "boxes" in target:
            target["boxes"] = boxes

        return PIL.Image.fromarray(img),target

class Resize(object):
    """
    适合先做pad（先按最长边填充成正方形），再做resize）
    也可以不pad，直接resize
    """
    def __init__(self,size=(224,224),multi_scale=False):
        self.size = size
        self.multi_scale = multi_scale
        if self.multi_scale: # 使用多尺度
            self.multi_scale_size = (32*np.arange(5,27,2)).tolist()

    def __call__(self, img,target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image: PIL image
                target: Tensor
        """
        if self.multi_scale:
            choice_size = random.choice(self.multi_scale_size)
            self.size = (choice_size, choice_size)

        img = np.asarray(img)
        original_size = img.shape[:2]
        # target["original_size"] = torch.as_tensor(original_size)
        target["resize"] = torch.as_tensor(self.size)

        # img = scipy.misc.imresize(img,self.size,'bicubic') #  or 'cubic'
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_CUBIC)

        if "boxes" in target:
            boxes = target["boxes"]

            boxes = self.resize_boxes(boxes,original_size,self.size)

            target["boxes"] = boxes
        return PIL.Image.fromarray(img), target

    def resize_boxes(self,boxes, original_size, new_size):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)

class Resize2(object):
    """先按比例resize，再pad"""
    def __init__(self,size=(224,224),multi_scale=False):
        self.size = size
        self.multi_scale = multi_scale
        if self.multi_scale:  # 使用多尺度
            self.multi_scale_size = (32 * np.arange(5, 27, 2)).tolist()

    def __call__(self, img,target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image: PIL image
                target: Tensor
        """
        if self.multi_scale:
            choice_size = random.choice(self.multi_scale_size)
            self.size = (choice_size,choice_size)

        img = np.asarray(img)
        img_h, img_w = img.shape[:2]

        target["original_size"] = torch.as_tensor((img_h, img_w))
        target["resize"] = torch.as_tensor(self.size)

        w, h = self.size
        new_w = int(img_w * min(w / img_w, h / img_h))
        new_h = int(img_h * min(w / img_w, h / img_h))

        if new_w >= new_h:
            new_w = max(new_w, w)
        else:
            new_h = max(new_h, h)

        # img = scipy.misc.imresize(img, [new_h,new_w], 'bicubic')  # or 'cubic'
        img = cv2.resize(img,(new_w,new_h),interpolation=cv2.INTER_CUBIC)

        if "boxes" in target:
            boxes = target["boxes"]
            boxes = Resize().resize_boxes(boxes, (img_h, img_w), (new_h,new_w))
            target["boxes"] = boxes

        # pad
        img,target = Pad().pad_img(img,target)

        return img,target

class RandomDrop(object):
    def __init__(self,p=0.5,cropsize=(0.1,0.1)):
        """cropsize:从原图裁剪掉的像素值范围比例"""
        self.cropsize = cropsize
        self.p = p

    def __call__(self,img,target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image: PIL image
                target: Tensor
        """
        try:
            _img, _target = self.do(img.copy(),target.copy())
            if len(_target["boxes"])>0:
                img, target=_img, _target
            del _img
            del _target
        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random() < self.p:
            img = np.asarray(img)
            img_h, img_w = img.shape[:2]
            y1,x1 = random.randint(0,int(self.cropsize[0]*img_h)),random.randint(0,int(self.cropsize[1]*img_w)) # 上面与左边裁剪的像素个数
            y2,x2 = random.randint(0,int(self.cropsize[0]*img_h)),random.randint(0,int(self.cropsize[1]*img_w)) # 下面与右边裁剪的像素个数

            # 裁剪后的图像
            img = img[y1:img_h-y2,x1:img_w-x2,:]
            new_h, new_w = img.shape[:2]

            # boxes也需做想要的裁剪处理
            boxes = target["boxes"]
            labels = target["labels"]
            new_boxes = []
            new_labels = []
            for i,b in enumerate(boxes):
                if b[2]-x1 <=0 or b[3]-y1 <=0: # box已经在裁剪图像的外
                    continue
                else:
                    new_boxes.append([max(0,b[0]-x1), max(0,b[1]-y1), min(new_w,b[2]-x1), min(new_h,b[3]-y1)])
                    new_labels.append(labels[i])

            new_boxes = torch.as_tensor(new_boxes)
            new_labels = torch.as_tensor(new_labels)
            target["boxes"] = new_boxes
            target["labels"] = new_labels

            return PIL.Image.fromarray(img),target
        else:
            return img,target

class RandomCrop(object):
    def __init__(self,p=0.5):
        self.p = p

    def __call__(self,img,target):
        """
        :param image: PIL image
        :param target: Tensor
        :return:
                image: PIL image
                target: Tensor
        """
        try:
            _img, _target = self.do(img.copy(), target.copy())
            if len(_target["boxes"]) > 0:
                img, target = _img, _target
            del _img
            del _target
        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random() < self.p:
            bgr = np.asarray(img)
            boxes = target["boxes"]  # .numpy()
            labels = target["labels"]  # .numpy()
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if (len(boxes_in) == 0):
                target["boxes"] = boxes
                target["labels"] = labels
                img = PIL.Image.fromarray(bgr)
                return img, target
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y + h, x:x + w, :]

            target["boxes"] = boxes_in
            target["labels"] = labels_in

            return PIL.Image.fromarray(img_croped),target
        else:
            return img,target

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img,target):
        try:
            _img, _target = self.do(img.copy(), target.copy())
            if len(_target["boxes"]) > 0:
                img, target = _img, _target
            del _img
            del _target
        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random() < self.p:
            img = np.asarray(img)
            img_center = np.array(img.shape[:2])[::-1] / 2
            img_center = np.hstack((img_center, img_center))
            img_center = torch.as_tensor(img_center,dtype=torch.float32)
            bboxes = target["boxes"]

            img = img[:, ::-1, :]
            bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])

            box_w = abs(bboxes[:, 0] - bboxes[:, 2])

            bboxes[:, 0] -= box_w
            bboxes[:, 2] += box_w

            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)

            target["boxes"]=bboxes

            return PIL.Image.fromarray(img), target
        else:
            return img,target

class RandomVerticallyFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img,target):
        try:
            _img, _target = self.do(img.copy(), target.copy())
            if len(_target["boxes"]) > 0:
                img, target = _img, _target
            del _img
            del _target
        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random() < self.p:
            img = np.asarray(img)
            img_center = np.array(img.shape[:2])[::-1] / 2
            img_center = np.hstack((img_center, img_center))
            img_center = torch.as_tensor(img_center,dtype=torch.float32)
            bboxes = target["boxes"]

            img = img[::-1, :, :]
            bboxes[:, [1, 3]] += 2 * (img_center[[1, 3]] - bboxes[:, [1, 3]])

            box_h = abs(bboxes[:, 1] - bboxes[:, 3])

            bboxes[:, 1] -= box_h
            bboxes[:, 3] += box_h

            target["boxes"]=bboxes

            return PIL.Image.fromarray(img), target
        else:
            return img, target

class RandomScale(object):
    def __init__(self, p = 0.5,scale=0.2, diff=False):
        self.scale = scale
        self.p = p

        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"
        else:
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)

        self.diff = diff

    def __call__(self, img, target):
        try:
            _img, _target = self.do(img.copy(), target.copy())
            if len(_target["boxes"]) > 0:
                img, target = _img, _target
            del _img
            del _target

        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random() < self.p:
            # Chose a random digit to scale by
            img = np.asarray(img)
            bboxes = target["boxes"].numpy()
            img_shape = img.shape

            if self.diff:
                scale_x = random.uniform(*self.scale)
                scale_y = random.uniform(*self.scale)
            else:
                scale_x = random.uniform(*self.scale)
                scale_y = scale_x

            resize_scale_x = 1 + scale_x
            resize_scale_y = 1 + scale_y

            img = cv2.resize(img, None, fx=resize_scale_x, fy=resize_scale_y)

            bboxes[:, :4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]

            canvas = np.zeros(img_shape, dtype=np.uint8)

            y_lim = int(min(resize_scale_y, 1) * img_shape[0])
            x_lim = int(min(resize_scale_x, 1) * img_shape[1])

            canvas[:y_lim, :x_lim, :] = img[:y_lim, :x_lim, :]

            img = canvas
            bboxes = clip_box(bboxes, [0, 0, 1 + img_shape[1], img_shape[0]], 0.25)

            target["boxes"] = torch.as_tensor(bboxes,dtype=torch.float32)
            img = PIL.Image.fromarray(img)
        return img, target

class RandomScale2(object):
    # #固定住高度，以0.8-1.2伸缩宽度，做图像形变
    def __init__(self, p = 0.5,scale=[0.8,1.2]):
        self.scale = random.uniform(*scale)
        self.p = p

    def __call__(self, img, target):
        try:
            _img, _target = self.do(img.copy(), target.copy())
            if len(_target["boxes"]) > 0:
                img, target = _img, _target
            del _img
            del _target
        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random() < self.p:
            # Chose a random digit to scale by
            img = np.asarray(img)
            height, width, c = img.shape
            boxes = target["boxes"]

            img = cv2.resize(img, (int(width * self.scale), height))
            scale_tensor = torch.FloatTensor([[self.scale, 1, self.scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor

            target["boxes"] = torch.as_tensor(boxes,dtype=torch.float32)
            img = PIL.Image.fromarray(img)

        return img, target

class RandomTranslate(object):
    """Randomly Translates the image """

    def __init__(self,p=0.5, translate=0.2, diff=False):
        self.translate = translate
        self.p = p

        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"
            assert self.translate[0] > 0 & self.translate[0] < 1
            assert self.translate[1] > 0 & self.translate[1] < 1


        else:
            assert self.translate > 0 and self.translate < 1
            self.translate = (-self.translate, self.translate)

        self.diff = diff

    def __call__(self, img, target):
        try:
            _img, _target = self.do(img.copy(), target.copy())
            if len(_target["boxes"]) > 0:
                img, target = _img, _target
            del _img
            del _target
        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random()<self.p:
            # Chose a random digit to scale by
            img = np.asarray(img)
            bboxes = target["boxes"].numpy()
            img_shape = img.shape

            # translate the image

            # percentage of the dimension of the image to translate
            translate_factor_x = random.uniform(*self.translate)
            translate_factor_y = random.uniform(*self.translate)

            if not self.diff:
                translate_factor_y = translate_factor_x

            canvas = np.zeros(img_shape).astype(np.uint8)

            corner_x = int(translate_factor_x * img.shape[1])
            corner_y = int(translate_factor_y * img.shape[0])

            # change the origin to the top-left corner of the translated box
            orig_box_cords = [max(0, corner_y), max(corner_x, 0), min(img_shape[0], corner_y + img.shape[0]),
                              min(img_shape[1], corner_x + img.shape[1])]

            mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]),
                   max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]), :]
            canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3], :] = mask
            img = canvas

            bboxes[:, :4] += [corner_x, corner_y, corner_x, corner_y]

            bboxes = clip_box(bboxes, [0, 0, img_shape[1], img_shape[0]], 0.25)

            target["boxes"] = torch.as_tensor(bboxes,dtype=torch.float32)

            return PIL.Image.fromarray(img), target
        else:
            return img,target


class RandomRotate(object):
    """Randomly rotates an image    """

    def __init__(self, p=0.5,angle=10):
        self.angle = angle
        self.p = p

        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"

        else:
            self.angle = (-self.angle, self.angle)

    def __call__(self, img, target):
        try:
            _img, _target = self.do(img.copy(), target.copy())
            if len(_target["boxes"]) > 0:
                img, target = _img, _target
            del _img
            del _target
        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random()<self.p:
            img = np.asarray(img)
            bboxes = target["boxes"].numpy()

            angle = random.uniform(*self.angle)

            w, h = img.shape[1], img.shape[0]
            cx, cy = w // 2, h // 2

            img = rotate_im(img, angle)

            corners = get_corners(bboxes)

            corners = np.hstack((corners, bboxes[:, 4:]))

            corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)

            new_bbox = get_enclosing_box(corners)

            scale_factor_x = img.shape[1] / w

            scale_factor_y = img.shape[0] / h

            img = cv2.resize(img, (w, h))

            new_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]

            bboxes = new_bbox

            bboxes = clip_box(bboxes, [0, 0, w, h], 0.25)

            target["boxes"] = torch.as_tensor(bboxes,dtype=torch.float32)

            return PIL.Image.fromarray(img), target
        else:
            return img,target


class RandomBrightness(object):
    def __init__(self,p=0.5,alpha=[0.5,1.5]):
        self.p = p
        self.alpha = alpha

    def __call__(self,img,target):
        try:
            _img, _target = self.do(img.copy(), target.copy())
            if len(_target["boxes"]) > 0:
                img, target = _img, _target
            del _img
            del _target
        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random() < self.p:
            img = np.asarray(img)
            hsv = BGR2HSV(img)
            h, s, v = cv2.split(hsv)
            adjust = random.choice(self.alpha)
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            img = HSV2BGR(hsv)
            img = PIL.Image.fromarray(img)

        return img, target


class RandomSaturation(object):
    def __init__(self,p=0.5,alpha=[0.5,1.5]):
        self.p = p
        self.alpha = alpha

    def __call__(self,img,target):
        try:
            _img, _target = self.do(img.copy(), target.copy())
            if len(_target["boxes"]) > 0:
                img, target = _img, _target
            del _img
            del _target
        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random() < self.p:
            bgr = np.asarray(img)
            hsv = BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice(self.alpha)
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = HSV2BGR(hsv)
            img = PIL.Image.fromarray(bgr)

        return img, target


class RandomHue(object):
    def __init__(self,p=0.5,alpha=[0.5,1.5]):
        self.p = p
        self.alpha = alpha

    def __call__(self,img,target):
        try:
            _img, _target = self.do(img.copy(), target.copy())
            if len(_target["boxes"]) > 0:
                img, target = _img, _target
            del _img
            del _target
        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random() < self.p:
            bgr = np.asarray(img)
            hsv = BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = HSV2BGR(hsv)
            img = PIL.Image.fromarray(bgr)

        return img, target


class RandomBlur(object):
    def __init__(self,p=0.5,kernel=(5,5)):
        self.p = p
        self.kernel = kernel

    def __call__(self,img,target):
        try:
            _img, _target = self.do(img.copy(), target.copy())
            if len(_target["boxes"]) > 0:
                img, target = _img, _target
            del _img
            del _target
        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random() < self.p:
            bgr = np.asarray(img)
            bgr = cv2.blur(bgr,self.kernel)
            img = PIL.Image.fromarray(bgr)

        return img, target


class RandomShift(object):
    # 平移变换 （有问题）
    def __init__(self,p=0.5):
        self.p = p

    def __call__(self,img,target):
        try:
            _img, _target = self.do(img.copy(), target.copy())
            if len(_target["boxes"]) > 0:
                img, target = _img, _target
            del _img
            del _target
        except:
            pass
        return img, target

    def do(self,img,target):
        if random.random() < self.p:
            bgr = np.asarray(img)
            boxes = target["boxes"]#.numpy()
            labels = target["labels"]#.numpy()
            center = (boxes[:, 2:] + boxes[:, :2]) / 2

            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)
            # print(bgr.shape,shift_x,shift_y)
            # 原图像的平移
            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),
                                                                     :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),
                                                                              :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,
                                                                             :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):,
                                                                                      -int(shift_x):, :]

            shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                target["boxes"] = boxes
                target["labels"] = labels
                img = PIL.Image.fromarray(bgr)
                return img, target
            box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(
                boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]

            target["target"] = boxes_in
            target["labels"] = labels_in
            img = PIL.Image.fromarray(after_shfit_image)

        return img, target


class ChannelMixer:
    """ Mix channels of multiple inputs in a single output image.
    This class works with opencv_ images (np.ndarray), and will mix the channels of multiple images into one new image.

    Args:
        num_channels (int, optional): The number of channels the output image will have; Default **3**

    Example:
        >>> # Replace the 3th channel of an image with a channel from another image
        >>> mixer = brambox.transforms.ChannelMixer()
        >>> mixer.set_channels([(0,0), (0,1), (1,0)])
        >>> out = mixer(img1, img2)
        >>> # out => opencv image with channels: [img0_channel0, img0_channel1, img1_channel0]
    """
    def __init__(self, num_channels=3):
        self.num_channels = num_channels
        self.channels = [(0, i) for i in range(num_channels)]

    def set_channels(self, channels):
        """ Set from which channels the output image should be created.
        The channels list should have the same length as the number of output channels.

        Args:
            channels (list): List of tuples containing (img_number, channel_number)
        """
        if len(channels) != self.num_channels:
            raise ValueError('You should have one [image,channel] per output channel')
        self.channels = [(c[0], c[1]) for c in channels]

    def __call__(self, *imgs):
        """ Create and return output image.

        Args:
            *imgs: Argument list with all the images needed for the mix

        Warning:
            Make sure the images all have the same width and height before mixing them.
        """
        m = max(self.channels, key=lambda c: c[0])[0]
        if m >= len(imgs):
            raise ValueError('{} images are needed to perform the mix'.format(m))

        if isinstance(imgs[0], Image.Image):
            pil_image = True
            imgs = [np.array(img) for img in imgs]
        else:
            pil_image = False

        res = np.zeros([imgs[0].shape[0], imgs[0].shape[1], self.num_channels], 'uint8')
        for i in range(self.num_channels):
            if imgs[self.channels[i][0]].ndim >= 3:
                res[..., i] = imgs[self.channels[i][0]][..., self.channels[i][1]]
            else:
                res[..., i] = imgs[self.channels[i][0]]
        res = np.squeeze(res)

        if pil_image:
            return Image.fromarray(res)
        else:
            return res

# ----------------------------------------------------------------------------

class RandomChoice(object):
    def __init__(self):
        pass

    def __call__(self,img,target):
        choice = random.choice([
            RandomCrop(),
            RandomScale2(),
            RandomScale(),
            RandomDrop(cropsize=(0.05, 0.05)),
        ])

        return choice(img,target)