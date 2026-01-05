import torch
import torch.nn as nn

class IoULoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.clamp(pred, 1e-6, 1-1e-6)
        target = (target > 0.1).float()

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection

        return 1 - (intersection + 1e-6) / (union + 1e-6)
