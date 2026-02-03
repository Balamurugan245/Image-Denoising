import os
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from Config import Config
from dataset import DenoiseDataset
from Unet import UNet
from loss import DenoisingLoss  


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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


class Trainer:
    def __init__(self, device, checkpoint_dir="checkpoints", pred_dir="predictions"):
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.pred_dir = pred_dir

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

        self.criterion = DenoisingLoss(
            lambda_edge=0.15,
            lambda_grad=0.02,
            lambda_bce=0.30
        ).to(device)

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
                    logits = model(noisy)                 # raw output
                    loss = self.criterion(logits, clean)  # expects logits

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

            # Save checkpoint
            torch.save(
                model.state_dict(),
                os.path.join(self.checkpoint_dir, f"model_epoch_{epoch:02d}.pth")
            )

            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                fixed_logits = model(fixed_noisy)
                fixed_pred = torch.sigmoid(fixed_logits).clamp(0, 1)

            epoch_dir = os.path.join(self.pred_dir, f"epoch_{epoch:02d}")
            os.makedirs(epoch_dir, exist_ok=True)

            save_triplet(
                fixed_noisy[0],
                fixed_clean[0],
                fixed_pred[0],
                os.path.join(epoch_dir, "comparison.png")
            )


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", DEVICE)

    dataset = DenoiseDataset(
        root=Config.data_root,
        img_size=384,               
        invert_target=True,          
        augment=True
    )

    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(DEVICE == "cuda")
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE == "cuda")
    )

    model = UNet(in_channels=1, out_channels=1).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.epochs,
        eta_min=1e-6
    )

    trainer = Trainer(device=DEVICE)
    trainer.start(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=Config.epochs
    )

if __name__ == "__main__":
    main()
