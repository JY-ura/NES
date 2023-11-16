import numpy as np
from torch.nn import functional as F
import torch
# from abc import ABC, abstractmethod
import torch
torch.optim.lr_scheduler.CosineAnnealingLR

__all__ = ['cos_scheduler', 'Momentum']
default_device = torch.device(0)


def cross_entorpy_loss(score, target_labels, targeted: str=False):
    target_labels = torch.argmax(target_labels, axis=-1)
    if targeted == 'untargeted':
        return -F.cross_entropy(score, target_labels, reduce=False)
    else:
        return F.cross_entropy(score, target_labels, reduce=False) # compute cross entorpy loss w.r.t. each sample

def margin_loss(score: torch.Tensor, target_labels: torch.Tensor, targeted: str):
    """compute margin loss, i.e. score[not target]_max - score[target], if <0, then we set it to 0 and say the attack is complete

    Args:
        score (torch.Tensor): _description_
        target_labels (torch.Tensor): _description_
    """
    target_score,_ = torch.max(score * target_labels, axis=-1)
    non_target_score, _ = torch.max(score * (1-1000 * target_labels), axis=-1)
    if targeted == 'targeted':
        return torch.maximum(torch.tensor(0.0), torch.log(non_target_score + 1e-6) - torch.log(target_score + 1e-6)).reshape(-1)
    else:
        return torch.maximum(torch.tensor(0.0), torch.log(target_score + 1e-6) - torch.log(non_target_score + 1e-6)).reshape(-1)


    """
    name: coslr
    max_lr: ${setup.optimizer.max_lr}
    min_lr: ${setup.optimizer.min_lr}  
    num_iter: 64
    warmup_iter_num: 5
    """

def cos_scheduler_increase(max_lr, min_lr, num_iter, warmup_iter_num, *args, **kwargs):
    """cos learning rate schedulr

    Args:
        max_lr (_type_): max learning rate
        min_lr (_type_): min learning rate
        num_iter (_type_): total num of iteration (i.e. batch num, epoch num .. etc.)
        warmup_iter_num (_type_): number of iters to use warmup

    Yields:
        _type_: _description_
    """
    for current_step in range(warmup_iter_num):
        yield min_lr + (max_lr - min_lr) * current_step / warmup_iter_num
    for current_step in range(num_iter - warmup_iter_num):
        current_step += warmup_iter_num
        curr_learning_rate = 0.5 * (1 + np.cos(current_step * np.math.pi / num_iter)) * max_lr
        yield np.maximum(curr_learning_rate, min_lr)


def cos_scheduler(max_lr, min_lr, num_iter, warmup_iter_num, *args, **kwargs):
    """Cosine learning rate scheduler with oscillation

    Args:
        max_lr (float): Maximum learning rate
        min_lr (float): Minimum learning rate
        num_iter (int): Total number of iterations (e.g., batch number, epoch number, etc.)
        warmup_iter_num (int): Number of iterations to use warmup

    Yields:
        float: Learning rate at each iteration
    """
    for current_step in range(warmup_iter_num):
        yield min_lr + (max_lr - min_lr) * current_step / warmup_iter_num

    for current_step in range(num_iter - warmup_iter_num):
        current_step += warmup_iter_num
        curr_learning_rate = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * current_step / (num_iter - warmup_iter_num)))
        yield curr_learning_rate

    # Continue oscillating beyond num_iter
    while True:
        current_step += 1
        curr_learning_rate = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * current_step / (num_iter - warmup_iter_num)))
        yield curr_learning_rate

def loss_lr(**kwargs):
    pass 

def clwars(**kwargs):
    pass

def step_lr_scheduler(max_lr, min_lr=1e-6, drop_epoch=30, gamma=0.95):
    it = 0
    lr = max_lr
    while True:
        it += 1
        if it % drop_epoch == 0:
            lr *= gamma
        yield np.maximum(lr, min_lr)


def linear_lr_scheduer(max_lr, min_lr, num_iter, warmup_iter_num, *args, **kwargs):
    for current_step in range(warmup_iter_num):
        yield min_lr + (max_lr - min_lr) * current_step / warmup_iter_num
    for current_step in range(num_iter - warmup_iter_num):
        current_step += warmup_iter_num
        curr_learning_rate = (max_lr - min_lr) * (1 - current_step / num_iter) + min_lr
        yield np.maximum(curr_learning_rate, min_lr)

class Momentum:
    def __init__(self, variabels: torch.Tensor, momentum, device=None) -> None:
        """ a simple SGD optimzier

        Args:
            variabels (torch.Tensor): parameter to register for futuer update 
            max_lr (_type_): max learning rate for a cosine scheduler
            min_lr (_type_): min learnign rate for a cosine schdueler 
            num_iter (_type_): (max) number of iteration to update 
            warmup_iter_num (_type_): warmup number for the momutum 
        """
        if device is None:
            device = default_device
        self.state = torch.zeros_like(variabels, device=device)
        self.momentum = momentum

    def apply_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """apply gradient and update the state

        Args:
            grad (torch.Tensor): gradient of the variables 

        Returns:
            torch.Tensor: updated variabels
        """
        self.state = self.state * (self.momentum) + grad * (1-self.momentum)
        return self.state
    
# class GradientTransform(abc.ABC):
#     @abstractmethod
#     def step(grad: any):
#         pass


# class SGD(GradientTransform):
    # def __init__(self, params, lr=0.01, momentum=None) -> None:
    #     super().__init__()
    #     self.params = params
    #     if momentum:
    #         self.m = torch.zeros_like(params)
    #     self.momentum = momentum

    # def step(grad: any):
    #     if hasattr(self, 'm'):
    #         self.m = self.momentum * self.m + (1 - self.momentum) * grad
    #         self.params.add_(-lr * self.m)

class Adam:
    def __init__(self, params, lr=0.01, beta_1=0.9, beta_2=0.99, eps=1e-8) -> None:
        """_init_

        Args:
            params (_type_): _description_
            lr (float, optional): _description_. Defaults to 0.01.
            beta_1 (float, optional): _description_. Defaults to 0.9.
            beta_2 (float, optional): _description_. Defaults to 0.99.
            eps (_type_, optional): _description_. Defaults to 1e-8.
        """
        super().__init__()
        self.params = params
        self.m = torch.zeros_like(params)
        self.v = torch.zeros_like(params)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.step = torch.tensor(1)
        self.eps = torch.tensor(eps)
    
    def apply_gradient(self, grad):
        self.m = self.beta_1 * self.m + (1-self.beta_1) * grad
        self.v = self.beta_2 * self.v + (1-self.beta_2) * grad ** 2
        m_t =  self.m / (1 - self.beta_1 ** self.step)
        v_t = self.v / (1- self.beta_2 ** self.step)
        delta = m_t / (torch.sqrt(v_t) + self.eps)
        self.step += 1
        return delta


