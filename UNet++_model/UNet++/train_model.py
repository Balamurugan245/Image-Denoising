import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_unetpp import UNetPP
from dataset import DenoisingDataset
from config import Config
from ie import save_triplet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CADLoss(torch.nn.Module):

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
             [0, 0, 0],
             [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def charbonnier(self, pred, target):

        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))

    def edge_loss(self, pred, target):

        grad_pred_x = torch.nn.functional.conv2d(pred, self.sobel_x, padding=1)
        grad_pred_y = torch.nn.functional.conv2d(pred, self.sobel_y, padding=1)

        grad_gt_x = torch.nn.functional.conv2d(target, self.sobel_x, padding=1)
        grad_gt_y = torch.nn.functional.conv2d(target, self.sobel_y, padding=1)

        return torch.nn.functional.l1_loss(grad_pred_x, grad_gt_x) + \
               torch.nn.functional.l1_loss(grad_pred_y, grad_gt_y)

    def forward(self, pred, target):

        loss_charb = self.charbonnier(pred, target)
        loss_edge = self.edge_loss(pred, target)

        return self.w_charb * loss_charb + self.w_edge * loss_edge

def train():

    noisy_dir = Config.noisy_dir
    clean_dir = Config.clean_dir

    train_dataset = DenoisingDataset(noisy_dir, clean_dir, split="train")
    val_dataset = DenoisingDataset(noisy_dir, clean_dir, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False
    )

    model = UNetPP().to(DEVICE)

    criterion = CADLoss().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.lr
    )

    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    os.makedirs(Config.prediction_dir, exist_ok=True)

    best_val = float("inf")

    for epoch in range(Config.epochs):
        model.train()

        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs}")

        for noisy, clean in pbar:
            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)
            optimizer.zero_grad()
            pred = model(noisy)
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for noisy, clean in val_loader:

                noisy = noisy.to(DEVICE)
                clean = clean.to(DEVICE)

                pred = model(noisy)

                val_loss += criterion(pred, clean).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        torch.save(
            model.state_dict(),
            f"{Config.checkpoint_dir}/epoch_{epoch+1:03d}.pth"
        )
        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                model.state_dict(),
                f"{Config.checkpoint_dir}/best_model.pth"
            )

            print("Saved BEST model")
