import math
import sys
import time
import torch
from tqdm import tqdm
from torch.cuda import amp
from torch.nn import functional as F
import torchvision.models.detection.mask_rcnn
import random

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from . import utils
from .tools import batch

def train_one_epoch_back(model, optimizer, data_loader, device, epoch, print_freq=50):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_iters = min(1000, len(data_loader) - 1)
        warmup_factor = 1. / 1000

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # images = list(image.to(device) for image in images)
        images = batch(images).to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k not in ["path","masks"]} for t in targets]

        # loss_dict = model(images, targets)
        loss_dict = model(images, targets, True, False)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def get_lr_scheduler(data_loader,optimizer,method=1):
    warmup_iters = len(data_loader) - 1

    if method==0:
        warmup_iters = min(1000, warmup_iters)
        warmup_factor = 1. / 1000
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    elif method == 1:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(warmup_iters//3,1), gamma=0.8)
    elif method == 2:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                        [warmup_iters // 3, warmup_iters // 2, warmup_iters // 1.2],gamma=0.8)
    elif method == 3:
        # warmup_iters = len(data_loader.dataset)//5
        # warmup_iters = len(data_loader.dataset)
        # warmup_iters = int(warmup_iters / math.sqrt(warmup_iters))
        # if warmup_iters % 2 == 0: warmup_iters += 1

        lr_scheduler = utils.build_lr_scheduler_ultralytics(optimizer, 0, warmup_iters)
    elif method == 4:
        lr = optimizer.param_groups[0]["lr"]
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, warmup_iters, lr*0.1)


    return lr_scheduler

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50,
                    multi_scale=False,use_amp=False,scaler=None,choose_multi_scale_step=5,
                    use_lr_scheduler=True,method=1):
    """use_amp=True 使用混合精度训练 效果会差
    0:160,1：120，2:140 ，use_lr_scheduler=False 120
    """
    # if use_amp and scaler is None:
        # scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if use_lr_scheduler:
        lr_scheduler = get_lr_scheduler(data_loader,optimizer,method)


    imgsz_min, imgsz_max = 320, 640
    gs = 64
    grid_min, grid_max = imgsz_min // gs, imgsz_max // gs

    for idx,(images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # images = list(image.to(device) for image in images)
        # images = torch.stack(images, 0)
        images = batch(images)

        if multi_scale and idx%choose_multi_scale_step==0: # 使用多尺度训练
            img_size = random.randrange(grid_min, grid_max + 1) * gs
            sf = img_size / max(images.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / gs) * gs for x in images.shape[2:]]  # new shape (stretched to 32-multiple)
                images = F.interpolate(images, size=ns, mode='bilinear', align_corners=False)

                _target = []
                for targ in targets:
                    targ["boxes"] *= sf
                    targ["resize"] *= sf
                    _target.append(targ)
                targets = _target

            metric_logger.add_meter("img_size", img_size)

        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k not in ["path"]} for t in targets]

        # Runs the forward pass under autocast.
        with torch.cuda.amp.autocast(use_amp):
            loss_dict = model(images, targets, True) # img_size

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if not use_amp or scaler is None:
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        elif scaler is not None and use_amp:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger



def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device,coco=None,iou_types=None):# iou_types=["bbox","segm","keypoints"]
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if coco is None:
        coco = get_coco_api_from_dataset(data_loader.dataset)
    if iou_types is None:
        iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        # images = list(img.to(device) for img in images)
        # images = torch.stack(images, 0).to(device)
        images = batch(images).to(device)
        new_target = [
            {k: v.to(device) for k, v in targ.items() if k not in ["path", "boxes", "labels", "masks"]} for targ in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        # with torch.cuda.amp.autocast(use_amp):
        outputs = model(images, new_target, False,True)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


# --------------------------accumulate------------------------------------------
def train_one_epoch_accumulate(model, optimizer, data_loader, device, epoch,
                               print_freq=50,accumulate=1,img_size=512,multi_scale=False,use_amp=False,scaler=None):
    """
    Args:
        model:
        optimizer:
        data_loader:
        device:
        epoch:
        print_freq:
        accumulate:  accumulate = max(round(64 / batch_size), 1)
        img_size:
        multi_scale:
        use_amp:
        scaler:  scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    Returns:

    """

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch >= 0:
        # warmup_iters = len(data_loader.dataset)//5
        warmup_iters = len(data_loader.dataset)
        warmup_iters = math.ceil(warmup_iters / math.sqrt(warmup_iters))

        # lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        lr_scheduler = utils.build_lr_scheduler_ultralytics(optimizer, 0, warmup_iters)

    # if epoch == 0:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
    #     warmup_factor = 1.0 / 1000
    #     warmup_iters = min(1000, len(data_loader) - 1)
    #
    #     lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    #     accumulate = 1

    nb = len(data_loader)

    imgsz_min, imgsz_max = 320, 640
    gs = 64
    grid_min, grid_max = imgsz_min // gs, imgsz_max // gs

    for i,(images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # ni 统计从epoch0开始的所有batch数
        ni = i + nb * epoch  # number integrated batches (since train start)
        images = batch(images)

        if multi_scale:  # 使用多尺度训练
            if ni%accumulate==0:
                img_size = random.randrange(grid_min, grid_max + 1) * gs
            sf = img_size / max(images.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / gs) * gs for x in images.shape[2:]]  # new shape (stretched to 32-multiple)
                images = F.interpolate(images, size=ns, mode='bilinear', align_corners=False)

                _target = []
                for targ in targets:
                    targ["boxes"] *= sf
                    targ["resize"] *= sf
                    _target.append(targ)
                targets = _target

        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k not in ["path", "masks"]} for t in targets]

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        # Runs the forward pass under autocast.
        with torch.cuda.amp.autocast(use_amp):
            loss_dict = model(images, targets, True)  # img_size

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        losses *= 1. / accumulate  # scale loss

        if scaler is None:
            losses.backward()
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
        elif scaler is not None and use_amp:
            scaler.scale(losses).backward()
            # optimize
            # 每训练64张图片更新一次权重
            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()


        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

        if ni % accumulate == 0 and lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

    return metric_logger


@torch.no_grad()
def evaluate_accumulate(model, data_loader,device,coco=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if coco is None:
        coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        # images = list(img.to(device) for img in images)
        # images = torch.stack(images, 0).to(device)
        images = batch(images).to(device)
        new_target = [
            {k: v.to(device) for k, v in targ.items() if k not in ["path", "boxes", "labels", "masks"]} for targ in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images, new_target, False,True)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    """
    result_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list
    
    coco_mAP = result_info[0]
    voc_mAP = result_info[1]
    coco_mAR = result_info[8]

    tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss', 'train/loss', "learning_rate",
            "mAP@[IoU=0.50:0.95]", "mAP@[IoU=0.5]", "mAR@[IoU=0.50:0.95]"]

    for x, tag in zip(mloss.tolist() + [lr, coco_mAP, voc_mAP, coco_mAR], tags):
        tb_writer.add_scalar(tag, x, epoch)
    """

    return coco_evaluator



# ---------------------------------classify-----------------------------------------------------------------
def train_one_epoch_cls(model, optimizer, data_loader, device, epoch, print_freq=50,
                        multi_scale=False,use_amp=False,scaler=None,use_lr_scheduler=True,method=1):
    # if use_amp and scaler is None:
    #     scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    total_loss = 0.0
    preds = []
    trues = []
    num_datas = len(data_loader.dataset)
    lr_scheduler = None
    # if epoch == 0:
    # # if epoch%5 == 0:
    #     # warmup_iters = num_datas//3
    #     warmup_iters = math.ceil(num_datas/math.sqrt(num_datas))
    #
    #     # lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    #     lr_scheduler = utils.build_lr_scheduler_ultralytics(optimizer, 0, warmup_iters)
    if use_lr_scheduler:
        lr_scheduler = get_lr_scheduler(data_loader,optimizer,method)

    imgsz_min, imgsz_max = 320, 640
    gs = 64
    grid_min, grid_max = imgsz_min // gs, imgsz_max // gs

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):

        if multi_scale: # 使用多尺度训练
            img_size = random.randrange(grid_min, grid_max + 1) * gs
            sf = img_size / max(images.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / gs) * gs for x in images.shape[2:]]  # new shape (stretched to 32-multiple)
                images = F.interpolate(images, size=ns, mode='bilinear', align_corners=False)

                _target = []
                for targ in targets:
                    targ["boxes"] *= sf
                    targ["resize"] *= sf
                    _target.append(targ)
                targets = _target

        # images = torch.stack(images,0).to(device)
        # targets = [{k: v.to(device) for k, v in t.items() if k not in ["path"]} for t in targets]
        # targets = torch.stack([target["labels"] for target in targets], 0).to(device)
        images = images.to(device)
        targets = targets.to(device)

        # Runs the forward pass under autocast.
        with torch.cuda.amp.autocast(use_amp):
            loss_dict,_preds = model(images, targets, True)


        preds.extend(_preds)
        trues.extend(targets)

        losses = sum(loss for loss in loss_dict.values())

        total_loss += losses.item() # 统计loss
        losses = losses/images.size(0) # 求平均loss

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


        if scaler is not None and use_amp:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            # optimizer.zero_grad() #放前 放后都可以
            losses.backward()
            optimizer.step()
            optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # return metric_logger

    tarin_loss = total_loss / num_datas
    tarin_acc = (torch.eq(torch.tensor(preds), torch.tensor(trues)).sum().float() / num_datas).item()
    return tarin_loss,tarin_acc


@torch.no_grad()
def evaluate_cls(model, data_loader, device,use_amp=False):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    total_loss = 0.0
    preds = []
    trues = []
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        # images = torch.stack(images,0).to(device)
        # targets = [
        #     {k: v.to(device) for k, v in targ.items() if k not in ["path"]} for targ in targets]
        # targets = torch.stack([target["labels"] for target in targets], 0).to(device)

        images = images.to(device)
        targets = targets.to(device)

        # torch.cuda.synchronize()
        with torch.cuda.amp.autocast(use_amp):
            outputs = model(images, targets, False,True)

        total_loss += outputs["valid_loss"]
        preds.extend(outputs["preds"])
        trues.extend(targets)

    num_datas = len(data_loader.dataset)
    valid_loss = total_loss / num_datas
    valid_acc = (torch.eq(torch.tensor(preds),torch.tensor(trues)).sum().float()/num_datas).item()

    # torch.set_num_threads(n_threads)
    print("\nvalid_loss:%.5f valid_acc:%.5f\n"%(valid_loss,valid_acc))

    return valid_loss,valid_acc


# ----------------------------classify muil-gpu------------------------------------
def train_one_epoch_clsV2(model, optimizer, data_loader, device, epoch,print_freq=50,multi_scale=False,use_amp=False,scaler=None):
    # if use_amp and scaler is None:
    #     scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    # loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if utils.is_main_process():
        data_loader = tqdm(data_loader)

    for step, (images, targets) in enumerate(data_loader):
        # images = torch.stack(images, 0).to(device)
        # targets = [{k: v.to(device) for k, v in t.items() if k not in ["path"]} for t in targets]
        # targets = torch.stack([target["labels"] for target in targets], 0).to(device)
        images = images.to(device)
        targets = targets.to(device)

        # Runs the forward pass under autocast.
        with torch.cuda.amp.autocast(use_amp):
            loss_dict, _preds = model(images, targets, True)

        loss = sum(loss for loss in loss_dict.values())/images.size(0)

        loss = utils.reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 在进程0中打印平均loss
        if utils.is_main_process():
            if step%print_freq==0:
                data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if scaler is None:
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        elif scaler is not None and use_amp:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()


@torch.no_grad()
def evaluate_clsV2(model, data_loader, device,use_amp=False):
    model.eval()

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if utils.is_main_process():
        data_loader = tqdm(data_loader)

    for step, (images, targets) in enumerate(data_loader):
        # images = torch.stack(images,0).to(device)
        # targets = torch.stack([target["labels"] for target in targets], 0).to(device)
        images = images.to(device)
        targets = targets.to(device)

        with torch.cuda.amp.autocast(use_amp):
            pred = model(images, targets, False,True)

        sum_num += torch.eq(pred, targets).sum()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num = utils.reduce_value(sum_num, average=False)

    return sum_num.item()

# ----------------------------classify muil-gpu------------------------------------

def train_one_epoch_clsV3(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if utils.is_main_process():
        data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        loss = utils.reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 在进程0中打印平均loss
        if utils.is_main_process():
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()


@torch.no_grad()
def evaluate_clsV3(model, data_loader, device):
    model.eval()

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if utils.is_main_process():
        data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num = utils.reduce_value(sum_num, average=False)

    return sum_num.item()