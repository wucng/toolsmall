"""
- https://github.com/LiyuanLucasLiu/RAdam

optimizer:

SGD
Moment
NesterovMoment (nag) (推荐)
AdaGrad
RMSProp (推荐)
Adam  (推荐)
RAdam  (推荐)
"""
import numpy as np
import math
import torch
from torch.optim.optimizer import Optimizer, required
from typing import Any, Callable, Dict, Iterable, List, Set, Type, Union
from enum import Enum
from bisect import bisect_right

__all__ = ['AdamW','RAdam','PlainRAdam',"build_optimizer","build_lr_scheduler"]

class Optimizer1():
    def __init__(self,lr=1e-2,esp = 1e-7):
        self.lr = lr
        self.esp =esp

class SGD(Optimizer1):
    def __init__(self,lr=None):
        super(SGD,self).__init__()
        if lr is not None:
            self.lr = lr
    def __call__(self, x,dx,step=None):
        x -= self.lr*dx
        return x

class Moment(Optimizer1):
    def __init__(self,mu=0.99,v=0,lr=None):
        super(Moment,self).__init__()
        self.mu = mu
        self.v = v
        if lr is not None:
            self.lr = lr
    def __call__(self, x,dx,step=None):
        self.v = self.mu*self.v-self.lr*dx
        x += self.v
        return x

class NesterovMoment(Optimizer1): # nag
    def __init__(self,mu=0.99,v=0,lr=None):
        super(NesterovMoment,self).__init__()
        self.mu = mu
        self.v = v
        if lr is not None:
            self.lr = lr
    def __call__(self, x,dx,step=None):
        v_prev = self.v
        self.v = self.mu*self.v-self.lr*dx
        x += -self.mu*v_prev+(1+self.mu)*self.v
        return x

class AdaGrad(Optimizer1):
    def __init__(self,cache=0,lr=None):
        super(AdaGrad,self).__init__()
        self.cache = cache
        if lr is not None:
            self.lr = lr
    def __call__(self, x,dx,step=None):
        self.cache += dx**2
        x -= self.lr*dx/(np.sqrt(self.cache)+self.esp)
        return x

class RMSProp(Optimizer1):
    def __init__(self,cache=0,decay_rate=0.9,lr=None):
        super(RMSProp,self).__init__()
        self.cache = cache
        self.decay_rate = decay_rate
        if lr is not None:
            self.lr = lr
    def __call__(self, x,dx,step=None):
        self.cache = self.decay_rate*self.cache+(1-self.decay_rate)*dx**2
        x -= self.lr*dx/(np.sqrt(self.cache)+self.esp)
        return x

class Adam(Optimizer1):
    def __init__(self,beta1=0.9,beta2=0.995,m = 0,v = 0,lr=None):
        super(Adam,self).__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = m
        self.v = v
        if lr is not None:
            self.lr = lr
    def __call__(self, x,dx,step):
        self.m = self.beta1*self.m + (1-self.beta1)*dx
        self.v = self.beta2*self.v + (1-self.beta2)*dx**2
        self.m /= 1+self.beta1**step
        self.v /= 1+self.beta2**step
        # self.m /= 1 - self.beta1 ** step
        # self.v /= 1 - self.beta2 ** step

        x -= self.lr*self.m / (np.sqrt(self.v)+self.esp)
        return x

class RAdam1(Optimizer1):
    def __init__(self,beta1=0.9,beta2=0.995,m = 0,v = 0,lr=None):
        super(RAdam1,self).__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = m
        self.v = v
        self.lou_00 = 2/(1-beta2)-1

        if lr is not None:
            self.lr = lr
    def __call__(self, x,dx,step):
        self.m = self.beta1*self.m + (1-self.beta1)*dx
        self.v = self.beta2*self.v + (1-self.beta2)*dx**2
        self.m /= 1+self.beta1**step
        self.v /= 1+self.beta2**step
        # self.m /= 1 - self.beta1 ** step
        # self.v /= 1 - self.beta2 ** step
        lou = self.lou_00 -2*step*self.beta2**step/(1-self.beta2**step)
        if lou>4:
            r = np.sqrt(((lou-4)*(lou-2)*self.lou_00)/((self.lou_00-4)*(self.lou_00-2)*lou))
            x -= self.lr * r * self.m / (np.sqrt(self.v) + self.esp)
        else:
            x -= self.lr*self.m / (np.sqrt(self.v)+self.esp)

        return x

# -----------https://github.com/LiyuanLucasLiu/RAdam---------------------------------------

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=4e-5, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class PlainRAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(PlainRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] * math.sqrt(
                        (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif self.degenerated_to_sgd:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p_data_fp32.add_(-step_size, exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup=warmup)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss

# -----------------detectron2/solver/build.py----------------------------------

_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]

class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"

def _create_gradient_clipper(clip_type="value")-> _GradientClipper:
    """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    """
    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, 1.0, 2.0)

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, 1.0)

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(clip_type)]


def _generate_optimizer_class_with_gradient_clipping(
    optimizer_type: Type[torch.optim.Optimizer], gradient_clipper: _GradientClipper
) -> Type[torch.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """

    def optimizer_wgc_step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                gradient_clipper(p)
        super(type(self), self).step(closure)

    OptimizerWithGradientClip = type(
        optimizer_type.__name__ + "WithGradientClip",
        (optimizer_type,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip


def maybe_add_gradient_clipping(
     optimizer: torch.optim.Optimizer,clip_gradients:bool=False,clip_type:str="value"
) -> torch.optim.Optimizer:
    """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer instance of some type OptimizerType to become an instance
    of the new dynamically created class OptimizerTypeWithGradientClip
    that inherits OptimizerType and overrides the `step` method to
    include gradient clipping.

    Args:
        cfg: CfgNode
            configuration options
        optimizer: torch.optim.Optimizer
            existing optimizer instance

    Return:
        optimizer: torch.optim.Optimizer
            either the unmodified optimizer instance (if gradient clipping is
            disabled), or the same instance with adjusted __class__ to override
            the `step` method and include gradient clipping
    """
    if not clip_gradients:
        return optimizer
    grad_clipper = _create_gradient_clipper(clip_type)
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        type(optimizer), grad_clipper
    )
    optimizer.__class__ = OptimizerWithGradientClip
    return optimizer


def build_optimizer(model: torch.nn.Module,base_lr:float=2.5e-4,
                    weight_decay:float=1e-4,momentum:float=0.9,
                    clip_gradients:bool=False,clip_type:str="value"
                    ) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module in model.modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            # lr = cfg.SOLVER.BASE_LR
            lr = base_lr
            # weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if isinstance(module, norm_module_types):
                weight_decay = 0.0
            elif key == "bias":
                # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                # hyperparameters are by default exactly the same as for regular
                # weights.
                lr = base_lr * 1.0
                weight_decay = 1e-4

            # elif key.startswith('backbone') and value.requires_grad:
            elif 'backbone' in key and value.requires_grad:
                lr = base_lr * 0.1 # 0.05
                weight_decay = 1e-5

            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # optimizer = torch.optim.SGD(params, base_lr, momentum=momentum)
    optimizer = torch.optim.RMSprop(params, base_lr, momentum=momentum)
    # optimizer = torch.optim.AdamW(params, base_lr, weight_decay=5e-4)
    optimizer = maybe_add_gradient_clipping(optimizer,clip_gradients,clip_type)
    return optimizer

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iters
        # instead of at 0. In the case that warmup_iters << max_iters the two are
        # very close to each other.
        return [
            base_lr
            * warmup_factor
            * 0.5
            * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()

def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,lr_scheduler_name="WarmupMultiStepLR",
    scale = 1
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = lr_scheduler_name
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            [210000//scale, 250000//scale],
            0.1,
            warmup_factor=1e-3,
            warmup_iters=1000//scale,
            warmup_method="linear",
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            300//scale,
            warmup_factor=1e-3,
            warmup_iters=1000//scale,
            warmup_method="linear",
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))


def build_optimizer2(model: torch.nn.Module,base_lr:float=2.5e-4,
                    weight_decay:float=1e-4,momentum:float=0.9) -> torch.optim.Optimizer:
    # construct an optimizer
    # params = [p for p in self.network.parameters() if p.requires_grad]
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'backbone' in key:
                params += [{'params': [value], 'lr': base_lr * 0.1, 'weight_decay': weight_decay*0.1}]
            else:
                params += [{'params': [value], 'lr': base_lr, 'weight_decay': weight_decay}]
    # optimizer = torch.optim.SGD(params, lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = torch.optim.RMSprop(params,base_lr,momentum=momentum, weight_decay=weight_decay)
    # optimizer = torch.optim.AdamW(params, base_lr, weight_decay=weight_decay)

    return optimizer

# ------------------https://github.com/ultralytics/yolov3---------------------------------------------
import math

def build_lr_scheduler_ultralytics(optimizer,start_epoch=0,epochs=100):
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1  # see link below
    # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
    return scheduler

def build_optimzer_ultralytics(model,hyp={},useAdam=False):
    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if useAdam:
        # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = torch.optim.Adam(pg0, lr=hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = torch.optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    return optimizer
