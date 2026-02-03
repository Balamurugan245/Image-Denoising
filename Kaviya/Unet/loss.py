import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        target = target.to(dtype=pred.dtype, device=pred.device)
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred, target):
        target = target.to(dtype=pred.dtype, device=pred.device)
        sobel_x = self.sobel_x.to(dtype=pred.dtype, device=pred.device)
        sobel_y = self.sobel_y.to(dtype=pred.dtype, device=pred.device)

        grad_pred_x = F.conv2d(pred, sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, sobel_y, padding=1)

        grad_gt_x = F.conv2d(target, sobel_x, padding=1)
        grad_gt_y = F.conv2d(target, sobel_y, padding=1)

        return F.l1_loss(grad_pred_x, grad_gt_x) + F.l1_loss(grad_pred_y, grad_gt_y)


class GradientLoss(nn.Module):
    def forward(self, pred, target):
        target = target.to(dtype=pred.dtype, device=pred.device)

        dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

        dx_gt = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        dy_gt = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        return F.l1_loss(dx_pred, dx_gt) + F.l1_loss(dy_pred, dy_gt)


class BoundaryWeightedBCELoss(nn.Module):
    def __init__(self, edge_weight=3.0):
        super().__init__()
        self.edge_weight = edge_weight

        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, logits, target):
        target = target.to(dtype=logits.dtype, device=logits.device)

        sobel_x = self.sobel_x.to(dtype=logits.dtype, device=logits.device)
        sobel_y = self.sobel_y.to(dtype=logits.dtype, device=logits.device)

        gx = F.conv2d(target, sobel_x, padding=1)
        gy = F.conv2d(target, sobel_y, padding=1)

        edge_map = torch.sqrt(gx * gx + gy * gy)
        edge_map = (edge_map > 0).to(dtype=logits.dtype)

        weight = 1.0 + self.edge_weight * edge_map

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (bce * weight).mean()


class DenoisingLoss(nn.Module):
    def __init__(self, lambda_edge=0.15, lambda_grad=0.02, lambda_bce=0.30):
        super().__init__()
        self.char = CharbonnierLoss()
        self.edge = EdgeLoss()
        self.grad = GradientLoss()
        self.bce  = BoundaryWeightedBCELoss(edge_weight=3.0)

        self.lambda_edge = lambda_edge
        self.lambda_grad = lambda_grad
        self.lambda_bce  = lambda_bce

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)

        loss_char = self.char(prob, target)
        loss_edge = self.edge(prob, target)
        loss_grad = self.grad(prob, target)
        loss_bce  = self.bce(logits, target)

        return loss_char + self.lambda_edge * loss_edge + self.lambda_grad * loss_grad + self.lambda_bce * loss_bce
