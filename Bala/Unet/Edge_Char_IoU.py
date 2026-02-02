import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeAwareIoULoss(nn.Module):
    def __init__(self, alpha=0.7, eps=1e-3):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def charbonnier(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))

    def sobel_edges(self, x):
        sobel_x = torch.tensor(
            [[[-1,0,1],[-2,0,2],[-1,0,1]]],
            dtype=x.dtype, device=x.device
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1,-2,-1],[0,0,0],[1,2,1]]],
            dtype=x.dtype, device=x.device
        ).unsqueeze(0)

        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)
      
    def forward(self, logits, target):
        pred = torch.sigmoid(logits)

        pred_edge = self.sobel_edges(pred)
        tgt_edge = self.sobel_edges(target)

        inter = (pred * target * tgt_edge).sum(dim=(1,2,3))
        union = ((pred + target) * tgt_edge).sum(dim=(1,2,3)) - inter
        iou_loss = 1 - ((inter + 1e-6) / (union + 1e-6)).mean()

        charb_loss = self.charbonnier(pred, target)

        return self.alpha * iou_loss + (1 - self.alpha) * charb_loss
