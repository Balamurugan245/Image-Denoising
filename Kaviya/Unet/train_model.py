import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


def save_triplet(noisy, clean, pred, save_path, invert_target=True):
    noisy = noisy.detach().cpu().squeeze(0)
    clean = clean.detach().cpu().squeeze(0)
    pred  = pred.detach().cpu().squeeze(0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(noisy, cmap="gray"); axes[0].set_title("Noisy Input"); axes[0].axis("off")
    axes[1].imshow(clean, cmap="gray"); axes[1].set_title("Ground Truth"); axes[1].axis("off")
    axes[2].imshow(pred,  cmap="gray"); axes[2].set_title("Denoised Output"); axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


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

        return loss_char + self.lambda_edge*loss_edge + self.lambda_grad*loss_grad + self.lambda_bce*loss_bce


class Trainer:
    def __init__(self, device, checkpoint_dir="checkpoints", pred_dir="predictions", invert_target=True):
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.pred_dir = pred_dir
        self.invert_target = invert_target

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

        self.criterion = DenoisingLoss(lambda_edge=0.15, lambda_grad=0.02, lambda_bce=0.30).to(device)

    def start(self, model, train_loader, val_loader, optimizer, scheduler, epochs):
        scaler = torch.amp.GradScaler("cuda", enabled=(self.device == "cuda"))

        fixed_noisy, fixed_clean = next(iter(val_loader))
        fixed_noisy = fixed_noisy.to(self.device)
        fixed_clean = fixed_clean.to(self.device)

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]")
            for noisy, clean in pbar:
                noisy = noisy.to(self.device, non_blocking=True)
                clean = clean.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                    logits = model(noisy)                 # raw logits
                    loss = self.criterion(logits, clean)  # criterion expects logits

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss /= len(train_loader)

            if scheduler is not None:
                scheduler.step()

            model.eval()
            val_loss = 0.0
            with torch.inference_mode():
                for noisy, clean in val_loader:
                    noisy = noisy.to(self.device, non_blocking=True)
                    clean = clean.to(self.device, non_blocking=True)

                    logits = model(noisy)
                    val_loss += self.criterion(logits, clean).item()

            val_loss /= len(val_loader)
            print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            torch.save(model.state_dict(), os.path.join(self.checkpoint_dir, f"model_epoch_{epoch:02d}.pth"))

            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                fixed_logits = model(fixed_noisy)
                fixed_pred = torch.sigmoid(fixed_logits).clamp(0, 1)

            epoch_dir = os.path.join(self.pred_dir, f"epoch_{epoch:02d}")
            os.makedirs(epoch_dir, exist_ok=True)
            save_triplet(
                fixed_noisy[0],
                fixed_clean[0],
                fixed_pred[0],
                os.path.join(epoch_dir, "comparison.png"),
                invert_target=self.invert_target
            )
