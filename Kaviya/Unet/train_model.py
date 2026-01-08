import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure
import matplotlib.pyplot as plt

from dataset import DenoisingDataset
from Unet import UNet


def ssim_loss(pred, target):
    return 1 - structural_similarity_index_measure(
        pred, target, data_range=1.0
    )


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_ROOT = "/kaggle/input/input-data/Dataset-1k/New_Data100"

    train_dataset = DenoisingDataset(
        f"{DATA_ROOT}/Noisy", f"{DATA_ROOT}/Clean", split="train"
    )
    val_dataset = DenoisingDataset(
        f"{DATA_ROOT}/Noisy", f"{DATA_ROOT}/Clean", split="val"
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    l1_loss = nn.L1Loss()

    def combined_loss(pred, target):
        return l1_loss(pred, target) + 0.5 * ssim_loss(pred, target)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)

    fixed_noisy, fixed_clean = next(iter(val_loader))
    fixed_noisy, fixed_clean = fixed_noisy.to(device), fixed_clean.to(device)

    EPOCHS = 40
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for noisy, clean in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            output = model(noisy)
            loss = combined_loss(output, clean)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                val_loss += combined_loss(model(noisy), clean).item()
                val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch}.pth")

        with torch.no_grad():
            pred = model(fixed_noisy)

        epoch_dir = f"predictions/epoch_{epoch}"
        os.makedirs(epoch_dir, exist_ok=True)
        save_image(fixed_noisy, f"{epoch_dir}/noisy.png")
        save_image(pred, f"{epoch_dir}/denoised.png")
        save_image(fixed_clean, f"{epoch_dir}/clean.png")


