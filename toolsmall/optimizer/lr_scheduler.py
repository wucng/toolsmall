from torch.optim.lr_scheduler import StepLR

# 主动去调用
class MyStepLR(StepLR):
    def __init__(self, optimizer,step_size=5,gamma=0.8,last_epoch=-1,last_lr=5e-5):
        self.last_lr = last_lr
        super(MyStepLR,self).__init__(optimizer,step_size,gamma,last_epoch)


    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [max(group['lr'],self.last_lr) for group in self.optimizer.param_groups]
        return [max(group['lr'] * self.gamma,self.last_lr)
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma,self.last_lr)
                for base_lr in self.base_lrs]


# 类似原来的StepLR 使用
class StepLRV2(StepLR):
    def __init__(self, optimizer, step_size=5, gamma=0.1, last_epoch=-1,last_lr=5e-5):
        self.last_lr = last_lr
        super(StepLRV2, self).__init__(optimizer, step_size, gamma, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [max(group['lr'],self.last_lr) for group in self.optimizer.param_groups]
        return [max(group['lr'] * self.gamma,self.last_lr)
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** (self.last_epoch // self.step_size),self.last_lr)
                for base_lr in self.base_lrs]
