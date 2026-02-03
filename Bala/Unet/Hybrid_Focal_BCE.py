import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCE(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class HybridCADLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.w_charb = 0.40
        self.w_edge  = 0.20
        self.w_iou   = 0.20
        self.w_bce   = 0.20

        self.eps = 1e-3
        self.bce = FocalBCE(alpha=0.8, gamma=2.0)

        sobel_x = torch.tensor(
            [[[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]], dtype=torch.float32
        ).unsqueeze(0)

        sobel_y = torch.tensor(
            [[[-1, -2, -1],
              [ 0,  0,  0],
              [ 1,  2,  1]]], dtype=torch.float32
        ).unsqueeze(0)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def charbonnier(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))

    def sobel(self, x):
        B, C, H, W = x.shape
        sx = self.sobel_x.repeat(C, 1, 1, 1)
        sy = self.sobel_y.repeat(C, 1, 1, 1)

        gx = F.conv2d(x, sx, padding=1, groups=C)
        gy = F.conv2d(x, sy, padding=1, groups=C)

        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def soft_iou(self, pred, target):
        inter = (pred * target).sum(dim=(1,2,3))
        union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter
        return 1 - ((inter + 1e-6) / (union + 1e-6)).mean()

    def forward(self, logits, target):
        pred = torch.sigmoid(logits)

        pred   = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        loss_charb = self.charbonnier(pred, target)
        loss_edge  = torch.mean(torch.abs(self.sobel(pred) - self.sobel(target)))
        loss_iou   = self.soft_iou(pred, target)
        loss_bce   = self.bce(logits, target)

        return (
            self.w_charb * loss_charb +
            self.w_edge  * loss_edge +
            self.w_iou   * loss_iou +
            self.w_bce   * loss_bce
        )
