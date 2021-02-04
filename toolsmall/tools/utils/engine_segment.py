from torch import nn
from . import utils_segment as utils
from .utils import build_lr_scheduler_ultralytics,reduce_dict
import math
import sys
import time
import torch


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            # image, target = image.to(device), target.to(device)
            image, target = torch.stack(image, 0).to(device), torch.stack(target, 0).to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        # image, target = image.to(device), target.to(device)
        image, target = torch.stack(image, 0).to(device), torch.stack(target, 0).to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

    return metric_logger


# ------------------------------------自定义-----------------------------------------
def train_one_epochV2(model, optimizer, data_loader, device, epoch, print_freq=50):
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
        lr_scheduler = build_lr_scheduler_ultralytics(optimizer, 0, warmup_iters)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images, targets = torch.stack(images,0).to(device), torch.stack(targets,0).to(device)
        loss_dict = model(images, targets, True)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
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


# @torch.no_grad()
def evaluateV2(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            # image, target = image.to(device), target.to(device)
            image, target = torch.stack(image, 0).to(device), torch.stack(target, 0).to(device)
            output = model(image,None,False)

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    print(confmat)

    return str(confmat).split("\n")