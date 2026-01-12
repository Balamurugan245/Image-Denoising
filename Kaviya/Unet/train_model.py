import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image
from torchmetrics.functional import structural_similarity_index_measure


def ssim_loss(pred, target):
    return 1 - structural_similarity_index_measure(pred, target, data_range=1.0)

import matplotlib.pyplot as plt

def save_triplet(noisy, clean, pred, save_path):
    noisy = noisy.detach().cpu().squeeze(0)
    clean = clean.detach().cpu().squeeze(0)
    pred  = pred.detach().cpu().squeeze(0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(noisy, cmap="gray")
    axes[0].set_title("Noisy Input")
    axes[0].axis("off")

    axes[1].imshow(clean, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Denoised Output")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

class Trainer:
    def __init__(self, device, checkpoint_dir="checkpoints", pred_dir="predictions"):
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.pred_dir = pred_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

        self.l1_loss = nn.L1Loss()

    def combined_loss(self, pred, target):
        return self.l1_loss(pred, target) + 0.5 * ssim_loss(pred, target)

    def start(self, model, train_loader, val_loader, optimizer, epochs):
        scaler = torch.amp.GradScaler("cuda", enabled=(self.device == "cuda"))

        fixed_noisy, fixed_clean = next(iter(val_loader))
        fixed_noisy = fixed_noisy.to(self.device)
        fixed_clean = fixed_clean.to(self.device)

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]")
            for noisy, clean in pbar:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                    output = model(noisy)
                    loss = self.combined_loss(output, clean)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.inference_mode():
                for noisy, clean in val_loader:
                    noisy = noisy.to(self.device)
                    clean = clean.to(self.device)
                    val_loss += self.combined_loss(model(noisy), clean).item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            torch.save(
                model.state_dict(),
                os.path.join(self.checkpoint_dir, f"model_epoch_{epoch:02d}.pth")
            )

            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                fixed_pred = model(fixed_noisy)
            epoch_dir = f"predictions/epoch_{epoch:02d}"
            os.makedirs(epoch_dir, exist_ok=True)
            save_triplet(
                fixed_noisy[0],
                fixed_clean[0],
                fixed_pred[0],
                f"{epoch_dir}/comparison.png")
