import sys, os
sys.path.append(os.path.abspath(".."))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import UNet
from src.dataset import DrawingDenoiseDataset
from src.loss_function import SSIMLoss
from src.train_pipeline import train_one_epoch

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = DrawingDenoiseDataset(
    noisy_dir="data/01-raw/noisy",
    clean_dir="data/01-raw/clean"
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

loss_fn = lambda p, t: 0.8 * nn.L1Loss()(p, t) + 0.2 * SSIMLoss()(p, t)

for epoch in range(200):
    loss = train_one_epoch(model, loader, optimizer, loss_fn, device)
    print(f"Epoch {epoch+1} | Loss: {loss:.4f}")

torch.save(model.state_dict(), "unet_denoiser.pth")
