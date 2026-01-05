import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import DrawingDenoiseDataset
from src.model import UNet
from src.loss_function import SSIMLoss
from src.utils import show_samples

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

dataset = DrawingDenoiseDataset(
    noisy_dir="data/train/noisy",
    clean_dir="data/train/clean"
)

loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
model = UNet().to(device)

l1_loss = nn.L1Loss()
ssim_loss = SSIMLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 200

for epoch in range(epochs):
    model.train()
    total_loss = 0
    start = time.time()

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        pred = model(noisy)

        loss_l1 = l1_loss(pred, clean)
        loss_ssim = ssim_loss(pred, clean)
        loss = 0.8 * loss_l1 + 0.2 * loss_ssim

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    epoch_time = time.time() - start
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(loader):.4f} | Time: {epoch_time:.2f}s")

show_samples(model, dataset, device, num=5)

torch.save(model.state_dict(), "unet_denoiser.pth")
