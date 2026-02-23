import torch
import torch.nn as nn
import torch.nn.functional as F

class CADLoss(nn.Module):

    def __init__(self, w_charb=1.0, w_edge=2.5, eps=1e-3):
        super().__init__()

        self.w_charb = w_charb
        self.w_edge = w_edge
        self.eps = eps

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def charbonnier(self, pred, target):
        return torch.mean(torch.sqrt((pred - target)**2 + self.eps**2))

    def edge_loss(self, pred, target):

        grad_pred_x = F.conv2d(pred, self.sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, self.sobel_y, padding=1)

        grad_gt_x = F.conv2d(target, self.sobel_x, padding=1)
        grad_gt_y = F.conv2d(target, self.sobel_y, padding=1)

        return F.l1_loss(grad_pred_x, grad_gt_x) + \
               F.l1_loss(grad_pred_y, grad_gt_y)

    def forward(self, pred, target):

        loss_charb = self.charbonnier(pred, target)
        loss_edge = self.edge_loss(pred, target)

        return self.w_charb * loss_charb + self.w_edge * loss_edge
