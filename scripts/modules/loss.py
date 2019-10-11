import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hard import get_device

class AlphaMatteLoss(nn.Module):
    def __init__(self):
        super(AlphaMatteLoss, self).__init__()
        self.device, _ = get_device()
        self.mse = nn.MSELoss()

    def forward(self, pred_alpha, pred_trimap, fg, bg, target_pile):
        pred_alpha = pred_alpha.where(pred_trimap < 0.66, torch.Tensor([1.]).to(self.device))
        pred_alpha = pred_alpha.where(pred_trimap > 0.33, torch.Tensor([0.]).to(self.device))
        pred_pile = fg * pred_alpha + bg * (1 - pred_alpha)
        return self.mse(pred_pile, target_pile)


class AmbiguousLoss(nn.Module):
    def __init__(self):
        super(AmbiguousLoss, self).__init__()

    def forward(self, pred_trimap):
        return 0.5 - torch.abs(pred_trimap - 0.5).mean()
