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

    def charbonnier(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))

    def sobel(self, x):
        sobel_x = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]],
                               device=x.device, dtype=x.dtype).unsqueeze(0)
        sobel_y = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]],
                               device=x.device, dtype=x.dtype).unsqueeze(0)
        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)
      
    def soft_iou(self, pred, target):
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()
        inter = (pred * target).sum(dim=(1,2,3))
        union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter
        return 1 - ((inter + 1e-6) / (union + 1e-6)).mean()

    def forward(self, logits, target):
        pred = torch.sigmoid(logits)

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

