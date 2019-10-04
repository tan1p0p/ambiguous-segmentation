import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, pred_alpha, fg, bg, target_pile):
        pred_pile = fg * pred_alpha + bg * (1 - pred_alpha)
        return F.mse_loss(pred_pile, target_pile)
