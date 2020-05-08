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

try:
    from .colormap import colormap
except:
    from colormap import colormap

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_RED = (20,50,255)
_WHITE = (255, 255, 255)

# 字在框上方
def vis_class(img, pos, class_str="person", font_scale=0.35,label=1,colors=[]):
    """Visualizes the class."""
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