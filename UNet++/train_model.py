import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_unetpp import UNetPP
from dataset import DenoisingDataset
#from loss import L1_Edge_Loss
from config import Config

import torchvision.transforms as T
#import pytorch_msssim
import torch.nn as nn
import torch as t
import torch.nn.functional as F
from typing import Tuple

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = (DEVICE == "cuda")

DATA_ROOT = r"C:\Users\ADMIN\Downloads\2kdata\2kdata\Dataset-1k\New_Data100"


noisy_dir = r"C:\Users\ADMIN\Downloads\2kdata\2kdata\Dataset-1k\New_Data100\Noisy"
clean_dir = r"C:\Users\ADMIN\Downloads\2kdata\2kdata\Dataset-1k\New_Data100\Clean"

train_dataset = DenoisingDataset(noisy_dir, clean_dir, split="train", val_ratio=0.2, channels=1)
val_dataset   = DenoisingDataset(noisy_dir, clean_dir, split="val", val_ratio=0.2, channels=1)

train_loader = DataLoader(train_dataset,
                          batch_size=Config.batch_size,
                          shuffle=True,
                          num_workers=0)

val_loader = DataLoader(val_dataset,
                        batch_size=Config.batch_size,
                        shuffle=False,
                        num_workers=0)

class Edge_IoU(t.nn.Module):
    def __init__(self, w1: float, w2: float, device: str) -> None:
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.device = device
        sobel_X = t.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=t.float32).view(1,1,3,3)
        sobel_Y = t.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=device, dtype=t.float32).view(1,1,3,3)
        self.register_buffer("sobel_X", sobel_X)
        self.register_buffer("sobel_Y", sobel_Y)

    def IOU(self, target: t.Tensor, pred: t.Tensor) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        # if not pred.requires_grad:
        #     raise ValueError("Predictions must have gradient tracking")
        target = t.where(target <= 0, t.ones_like(target, device=self.device), 
                            t.zeros_like(target, device=self.device))
        pred = t.sigmoid(pred)
        projected_target = target.view((target.size(0), -1))
        projected_pred = pred.view((target.size(0), -1))

        intersection = projected_target * projected_pred
        intersection_sum = intersection.sum(dim=1)
        union = projected_pred.sum(dim=1) + projected_target.sum(dim=1) - intersection_sum
        iou = (intersection_sum + 1e-5) / (union + 1e-5)
        iou_loss = 1 - iou
        return iou_loss.mean(), target, intersection.view((target.shape))
    
    def edge_loss(self, target, pred):
        t_x = F.conv2d(target, self.sobel_X, padding=1)
        p_x = F.conv2d(pred, self.sobel_X, padding=1)
        t_y = F.conv2d(target, self.sobel_Y, padding=1)
        p_y = F.conv2d(pred, self.sobel_Y, padding=1)
        edge_t = t.abs(t_x) + t.abs(t_y) + 1e-6
        edge_p = t.abs(p_x) + t.abs(p_y) + 1e-6
        e_loss = F.l1_loss(edge_p, edge_t)
        return e_loss
    
    def forward(self, target: t.Tensor, pred: t.Tensor) -> t.Tensor:
        iou_loss, target, pred = self.IOU(target, pred)
        e_loss = self.edge_loss(target, pred)
        loss = self.w1 * iou_loss + self.w2 * e_loss
        return loss
        
criterion = Edge_IoU(w1=0.7,w2=0.3,device=DEVICE).to(DEVICE)

model = UNetPP().to(DEVICE)
#criterion = L1_Edge_Loss(alpha=1.0, beta=0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)

import os
os.makedirs(Config.checkpoint_dir, exist_ok=True)
os.makedirs(Config.prediction_dir, exist_ok=True)

fixed_noisy_list, fixed_clean_list = [], []

for noisy, clean in val_loader:
    fixed_noisy_list.append(noisy)
    fixed_clean_list.append(clean)
    if len(fixed_noisy_list) == 3:
        break

fixed_noisy = torch.cat(fixed_noisy_list).to(DEVICE)
fixed_clean = torch.cat(fixed_clean_list).to(DEVICE)

scaler = torch.amp.GradScaler(enabled=(DEVICE=="cuda"))
train_losses, val_losses = [], []

best_val = float("inf")

def save_triplet(noisy, clean, pred, path):
    noisy = noisy.cpu().squeeze()
    clean = clean.cpu().squeeze()
    pred  = pred.cpu().squeeze()

    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].imshow(noisy, cmap="gray"); ax[0].set_title("Noisy")
    ax[1].imshow(clean, cmap="gray"); ax[1].set_title("Clean GT")
    ax[2].imshow(pred, cmap="gray");  ax[2].set_title("Prediction")
    for a in ax: a.axis("off")
    plt.savefig(path, dpi=200)
    plt.close()

for epoch in range(Config.epochs):

    model.train()
    train_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs}")
    for noisy, clean in pbar:
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE=="cuda")):
            pred = model(noisy)
            loss = criterion(clean, pred)  #due to new loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            pred = model(noisy)
            val_loss += criterion(clean, pred).item()

    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1} | Train {train_loss:.4f} | Val {val_loss:.4f}")

    torch.save(model.state_dict(), f"{Config.checkpoint_dir}/epoch_{epoch+1:03d}.pth")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), f"{Config.checkpoint_dir}/best_model.pth")
        print("Saved BEST model")

    with torch.no_grad():
        fixed_pred = model(fixed_noisy)
        fixed_pred = torch.sigmoid(fixed_pred)
        fixed_pred = torch.clamp(fixed_pred, 0, 1)
        fixed_pred = 1 - fixed_pred    #for maintain the white and black

    epoch_dir = f"{Config.prediction_dir}/epoch_{epoch+1:03d}"
    os.makedirs(epoch_dir, exist_ok=True)

    for i in range(3):
        save_triplet(
            fixed_noisy[i],
            fixed_clean[i],
            fixed_pred[i],
            f"{epoch_dir}/sample_{i}.png"
        )      