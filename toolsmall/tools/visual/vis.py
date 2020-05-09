"""
opencv 不支持中文，不能显示中文
"""
import cv2
import skimage.io as io
from PIL import Image,ImageDraw
import PIL.ImageDraw
import numpy as np
import PIL.Image
import os
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

# import sys
# sys.path.append("../detectron/utils")
try:
    from .colormap import colormap
except:
    from colormap import colormap

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_RED = (20,50,255)
_WHITE = (255, 255, 255)

# 字在框内和字在框上方
def vis_rect(img, pos,class_str="person",font_scale=0.35,label=1,colors=[],inside=True,useMask=True):
    """
    :param img: BRG格式
    :param pos: [x1,y1,x2,y2]
    :param class_str:
    :param font_scale:
    :param label:
    :param colors:[[B,G,R],[B,G,R],...]
    :param inside:True 字在框内；False 字在框上面
    :param useMask:True 使用mask；FALSE不使用mask
    :return:BRG格式
    """
    c1=pos[:2]
    c2=pos[2:4]

    font = cv2.FONT_HERSHEY_SIMPLEX  # cv2.FONT_HERSHEY_PLAIN
    t_size = cv2.getTextSize(class_str, font, font_scale, 1)[0]

    color = colors[label] if colors else colormap()[label].tolist()
    cv2.rectangle(img, (c1[0], c1[1]), (c2[0], c2[1]), color, 2)
    if useMask:
        # mask
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[c1[1]:c2[1],c1[0]:c2[0]]=1
        # color_mask = [255, 184,99]  # BGR
        color_mask = np.asarray(color, np.uint8).reshape([1, 3])
        img = vis_mask(img, mask, color_mask, 0.3, False) # True

    if inside:
        # 文字在框内
        back_tl = c1[0], c1[1]
        back_br = c1[0] + t_size[0]+6, c1[1]+ t_size[1] + 10
        txt_tl = c1[0]+5, c1[1] + t_size[1] + 4
    else:
        # 文字在框上面
        back_tl = c1[0], c1[1] - int(1.2 * t_size[1])
        back_br = c1[0] + t_size[0], c1[1]
        txt_tl = c1[0], c1[1] - int(0.2 * t_size[1])

    cv2.rectangle(img, back_tl, back_br, color, -1)
    cv2.putText(img, class_str, txt_tl, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
    return img

# 字在框上方
def vis_class(img, pos, class_str="person", font_scale=0.35,label=1,colors=[]):
    """Visualizes the class."""
    # temp_GREEN=np.clip(np.asarray(_GREEN)*label,0,255).astype(np.uint8).tolist()

    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.2 * txt_h)
    back_br = x0 + txt_w, y0
    color = colors[label] if colors else colormap()[label].tolist()
    cv2.rectangle(img, back_tl, back_br, color, -1) # _GREEN
    # Show text.
    txt_tl = x0, y0 - int(0.2 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
    cv2.rectangle(img,(pos[0],pos[1]),(pos[2],pos[3]),color,2) # _GREEN
    return img

def vis_mask(img, mask, col, alpha=0.4, show_border=True, border_thick=1):
    """Visualizes a single binary mask."""
    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col

    if show_border:
        try:
            _, contours, _ = cv2.findContours( # contours, _
                mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)
        except:
            contours, _ = cv2.findContours(  # _, contours, _
                mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)

def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    keypoint_flip_map = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }
    return keypoints, keypoint_flip_map

def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines

def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints, _ = get_keypoints()
    kp_lines = kp_connections(dataset_keypoints)

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_hip')] +
        kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_hip')],
        kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')
    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
            color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(mid_hip),
            color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def show_img(img,boxes1,boxes2,labels):
    img1=np.asarray(img,np.uint8).copy()
    for pos,label in zip(boxes1,labels):
        class_str = "%s" % (label)
        img1=vis_class(img1,list(map(int,pos)),class_str,0.5,label)

    img2 = np.asarray(img, np.uint8).copy()
    for pos, label in zip(boxes2, labels):
        class_str = "%s" % (label)
        img2 = vis_class(img2, list(map(int, pos)), class_str, 0.5, label)

    _,axis=plt.subplots(1,2)
    axis[0].imshow(img1)
    axis[0].set_title("img1")
    axis[1].imshow(img2)
    axis[1].set_title("img2")

    plt.show()

# 画框
def draw_rect(img,target,classes=[],colors=[]):
    boxes, labels=target["boxes"],target["labels"]
    img=np.asarray(img,np.uint8).copy()
    for pos,label in zip(boxes,labels):
        if sum(pos)==0.:continue
        class_str = "%s" % (label) if len(classes)==0 else "%s" % (classes[label])
        img=vis_class(img,list(map(int,pos)),class_str,0.5,label,colors)

    # plt.imshow(img)
    # plt.title("img")
    # plt.show()
    return img

def draw_mask(image,target,segms=True):
    """segms=False 不画mask"""
    image = np.asarray(image, np.uint8).copy()
    labels = target["labels"]
    bboxs = target["boxes"]
    scores = np.ones(len(labels),np.float32) # 虚假的

    if segms:
        masks = target["masks"]
        color_list = colormap()
        # color_list = colormap(rgb=True) / 255
        mask_color_id=0

    for idx,(label,bbox,score) in enumerate(zip(labels,bboxs,scores)):
        # label, bbox, score = label.cpu().numpy(), bbox.cpu().numpy(), score.cpu().numpy()
        class_str="%s:%.3f"%(label,score)
        pos=list(map(int,bbox))
        image=vis_class(image,pos,class_str,0.5,label)
        if segms:
            mask = masks[idx]#.cpu().numpy()
            mask = np.clip(mask * 255., 0, 255).astype(np.uint8) # np.squeeze(mask, 0)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1
            image=vis_mask(image,mask,color_mask)

    plt.imshow(image)
    plt.title("img")
    plt.show()
    # return image

def draw_segms(image,target):
    image = np.asarray(image, np.uint8).copy()
    masks = np.asarray(target, np.uint8).copy()
    color_list = colormap()

    unique_label = np.unique(masks)
    for i, la in enumerate(unique_label):
        if la == 0: continue  # 背景跳过
        mask = np.asarray(masks == la)
        mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
        # color_mask = color_list[i % len(color_list), 0:3]
        color_mask = color_list[la % len(color_list), 0:3]
        image = vis_mask(image, mask, color_mask)

    plt.imshow(image)
    plt.title("img")
    plt.show()
    # return image

def draw_keypoint(image,target,draw_bboxs=False,draw_keypoints=False):
    """
    :param image:
    :param target:
    :param draw_bboxs: False 不画bboxs
    :param draw_keypoints: False 不画keypoints
    :return:
    """
    image = np.asarray(image, np.uint8).copy()

    labels = target["labels"]
    bboxs = target["boxes"]
    scores = np.ones(len(labels),np.float32) # 虚假的

    if draw_keypoints:
        keypoints = target["keypoints"]
        keypoints_scores = np.ones([len(labels),17],np.float32)*3

    for idx,(label,bbox,score) in enumerate(zip(labels,bboxs,scores)):
        if draw_bboxs:
            # label, bbox, score = label.cpu().numpy(), bbox.cpu().numpy(), score.cpu().numpy()
            class_str="%s:%.3f"%(label,score)
            pos=list(map(int,bbox))
            image=vis_class(image,pos,class_str,0.5,label)
        if draw_keypoints:
            keypoint = keypoints[idx]#.cpu().numpy()
            keypoints_score = keypoints_scores[idx]#.cpu().numpy()
            # temp=np.hstack((keypoint,keypoints_score[:,None])).T  # shape (4, #keypoints) where 4 rows are (x, y, logit, prob)
            temp=np.hstack((keypoint,keypoints_score[:,None])).T  # shape (4, #keypoints) where 4 rows are (x, y, logit, prob)
            temp1=temp.copy()[-1]
            temp2=temp.copy()[-2]
            temp[-2]=temp1
            temp[-1]=temp2

            image = vis_keypoints(image,temp, 2)

    plt.imshow(image)
    plt.title("img")
    plt.show()
    # return image
