import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

class HybridCADLoss(nn.Module):
    def __init__(
        self,
        w_charb=0.45,
        w_edge=0.25,
        w_iou=0.10,
        w_bce=0.20,
        eps=1e-3
    ):
        super().__init__()
        self.w_charb = w_charb
        self.w_edge = w_edge
        self.w_iou = w_iou
        self.w_bce = w_bce
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def charbonnier(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))
      
    def sobel(self, x):
        sobel_x = torch.tensor(
            [[[-1,0,1],[-2,0,2],[-1,0,1]]],
            device=x.device, dtype=x.dtype
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1,-2,-1],[0,0,0],[1,2,1]]],
            device=x.device, dtype=x.dtype
        ).unsqueeze(0)

        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def soft_iou(self, pred, target):
        inter = (pred * target).sum(dim=(1,2,3))
        union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter
        return 1 - ((inter + 1e-6) / (union + 1e-6)).mean()
      
    def forward(self, logits, target):
        pred = torch.sigmoid(logits)

        loss_charb = self.charbonnier(pred, target)

        edge_pred = self.sobel(pred)
        edge_tgt  = self.sobel(target)
        loss_edge = torch.mean(torch.abs(edge_pred - edge_tgt))

        loss_iou = self.soft_iou(pred, target)

        loss_bce = self.bce(logits, target)

        return (
            self.w_charb * loss_charb +
            self.w_edge  * loss_edge +
            self.w_iou   * loss_iou +
            self.w_bce   * loss_bce
        )

