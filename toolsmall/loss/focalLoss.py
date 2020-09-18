# https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from torch.nn import functional as F
from fvcore.nn import (focal_loss,giou_loss,smooth_l1_loss,
                            sigmoid_focal_loss_jit,sigmoid_focal_loss_star_jit)
# from math import pi,atan

__all__=["giou_loss_jit","smooth_l1_loss_jit",
         "sigmoid_focal_loss_jit","sigmoid_focal_loss_star_jit",
         "diou_loss_jit","ciou_loss_jit","smooth_label_cross_entropy_loss_jit"]

giou_loss_jit = torch.jit.script(
    giou_loss
)  # type: torch.jit.ScriptModule


smooth_l1_loss_jit = torch.jit.script(
    smooth_l1_loss
)  # type: torch.jit.ScriptModule


def ciou_loss(
    boxes1: torch.Tensor,# [x1,y1,x2,y2]
    boxes2: torch.Tensor,# # [x1,y1,x2,y2]
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Generalized Intersection over Union Loss (Hamid Rezatofighi et. al)
    https://arxiv.org/abs/1902.09630

    Gradient-friendly IoU loss with an additional penalty that is non-zero when the
    boxes do not overlap and scales with the size of their smallest enclosing box.
    This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.

    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)

    # smallest enclosing box
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    # 对角线长度
    diag_len= (xc2-xc1)**2+(yc2-yc1)**2

    # 中心点距离平方
    x0 = (x1+x2)/2
    y0 = (y1+y2)/2
    x0g = (x1g+x2g)/2
    y0g = (y1g+y2g)/2
    center_len = (x0-x0g)**2+(y0-y0g)**2

    #
    w = x2-x1
    h = y2-y1
    wg = x2g-x1g
    hg = y2g-y1g
    pi = -4*torch.atan(torch.tensor(-1.,device=boxes1.device,dtype=torch.float32))
    v = 4/pi**2*(torch.atan(wg/hg)-torch.atan(w/h))**2
    alpha = v/(1-iouk+v)

    loss = 1 - iouk+center_len/diag_len+alpha*v

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

ciou_loss_jit = torch.jit.script(
    ciou_loss
)  # type: torch.jit.ScriptModule


def diou_loss(
    boxes1: torch.Tensor,# [x1,y1,x2,y2]
    boxes2: torch.Tensor,# # [x1,y1,x2,y2]
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Generalized Intersection over Union Loss (Hamid Rezatofighi et. al)
    https://arxiv.org/abs/1902.09630

    Gradient-friendly IoU loss with an additional penalty that is non-zero when the
    boxes do not overlap and scales with the size of their smallest enclosing box.
    This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.

    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)

    # smallest enclosing box
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    # 对角线长度
    diag_len= (xc2-xc1)**2+(yc2-yc1)**2

    # 中心点距离平方
    x0 = (x1+x2)/2
    y0 = (y1+y2)/2
    x0g = (x1g+x2g)/2
    y0g = (y1g+y2g)/2
    center_len = (x0-x0g)**2+(y0-y0g)**2

    loss = 1 - iouk+center_len/diag_len

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

diou_loss_jit = torch.jit.script(
    diou_loss
)  # type: torch.jit.ScriptModule


# smooth loss
def smooth_label_cross_entropy_loss(
        preds:torch.Tensor, # [N,num_classes]
        targets:torch.Tensor, # [N,]
        alpha:float = 0.03,
        reduction: str = "none",
        useSoftmax:bool=False
) -> torch.Tensor:
    """
    :param preds:
    :param targets:
    :param reduction:
    :return:
    """
    num_classes = preds.size(-1)
    onehot_label = F.one_hot(targets,num_classes).float().to(preds.device)
    # label smooth
    onehot_label = onehot_label * (1 - alpha) + alpha / num_classes * torch.ones_like(onehot_label)

    # cross entropy loss
    if useSoftmax:
        loss = F.binary_cross_entropy_with_logits(torch.softmax(preds,-1), onehot_label,reduction=reduction)
    else:
        loss = sigmoid_focal_loss_jit(preds, onehot_label, 0.2, 2.0, reduction=reduction)

    return loss

smooth_label_cross_entropy_loss_jit = torch.jit.script(
    smooth_label_cross_entropy_loss
)  # type: torch.jit.ScriptModule


# -------------------自定义 focal loss---------------------------
def sigmoid_focal_loss_binary(pred:torch.tensor,true:torch.tensor,alpha=0.25,gamma=2,reduction ="mean"):
    """
    p = sigmoid(p)
    CE(p) = -y*log(p)-(1-y)*log(1-p)
    """
    assert reduction in ["mean","sum"]
    assert pred.shape == true.shape

    pred = torch.sigmoid(pred)

    loss_positive = -(1-alpha)*(1-pred)**gamma*torch.log(pred)*true
    loss_negative = -alpha*pred**gamma*torch.log(1-pred)*(1-true)

    loss = loss_positive+loss_negative
    if reduction == "sum":
        loss = loss.sum()
    else:
        loss = loss.mean()

    return loss

def softmax_focal_loss_mullabels(pred:torch.tensor,true:torch.tensor,alpha=0.25,gamma=2,reduction ="mean"):
    """
    p = softmax(p)
    CE(p) = -y*log(p)

    or

    CE(p) = -y*log_softmax(p)
    """
    assert reduction in ["mean", "sum"]
    assert pred.shape == true.shape
    pred = torch.softmax(pred,-1)

    w = torch.ones_like(pred)
    w[..., 0] *= alpha
    w[..., 1:] *= 1 - alpha
    # 归一化
    w = torch.softmax(w, -1)

    loss = - w*(1-pred)**gamma*torch.log(pred)*true

    if reduction == "sum":
        loss = loss.sum()
    else:
        loss = loss.mean()

    return loss

if __name__=="__main__":
    # pred = torch.rand([5,])
    # y = torch.randint(0,2,[5,],dtype=torch.float32)

    # loss = sigmoid_focal_loss(pred,y,reduction ="mean")
    # loss = sigmoid_focal_loss_star(pred,y,reduction ="mean")
    # loss = sigmoid_focal_loss_jit(pred,y,reduction ="mean")
    # loss = sigmoid_focal_loss_star_jit(pred,y,reduction ="mean")

    # pred = torch.rand([5,10])
    # y = torch.randint(0, 10, [5, ], dtype=torch.long)

    # loss = softmax_focal_loss(pred,y,reduction ="mean")
    # loss = softmax_focal_loss_jit(pred,y,reduction ="mean")
    # loss = softmax_focal_loss_star(pred,y,reduction ="mean")
    # loss = softmax_focal_loss_star_jit(pred,y,reduction ="mean")


    # box1 = torch.as_tensor([[10,20,30,40],[120,180,189,200]])
    # box2 = torch.as_tensor([[15,23,30,46],[116,175,194,207]])
    # loss = giou_loss(box1,box2,reduction="mean")

    pred = torch.rand([5, 10])
    y = torch.rand([5, 10])
    loss = smooth_l1_loss_jit(pred,y,1e-3,reduction="mean")

    print(loss)