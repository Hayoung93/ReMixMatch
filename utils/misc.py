import numpy as np
import torch
from torch import nn


class SharpenSoftmax(nn.Module):
    def __init__(self, T, dim=0):
        super().__init__()
        self.T = T
        self.dim = dim
    
    def forward(self, pred):
        pred = pred ** (1 / self.T)
        return pred.softmax(self.dim)


class Div255(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, tensor):
        return tensor / 255.


class LogWeight():
    def __init__(self, exp, max_ep):
        self.line = (exp ** (np.asarray(list(range(max_ep))) / max_ep) - 1) / (exp - 1)
    
    def __call__(self, ep):
        return self.line[ep]


def get_tsa_mask(pred, max_epoch, epoch, iter_per_epoch, iteration):
    # Use linear TSA strategy
    max_iter = max_epoch * iter_per_epoch
    tsa_th = (epoch * iter_per_epoch + iteration + 1) / max_iter
    return pred.softmax(dim=1) <= tsa_th

def load_full_checkpoint(model, optimizer, scheduler, weight_path):
    cp = torch.load(weight_path)
    model.load_state_dict(cp["state_dict"])
    optimizer.load_state_dict(cp["optimizer"])
    scheduler.load_state_dict(cp["scheduler"])
    return model, optimizer, scheduler, cp["epoch"], cp["best_val_loss"], cp["best_val_acc"], cp["p_label"], cp["p_pred"]
