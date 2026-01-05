# train.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DenoiseDataset
from model import UNetCBAM
from loss import IoULoss
import config

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

dataset = DenoiseDataset(config.DATA_ROOT, config.IMG_SIZE)
loader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS
)

model = UNetCBAM().to(device)
criterion = IoULoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

for epoch in range(config.EPOCHS):
    model.train()
    running_loss = 0

    pbar = tqdm(loader, desc=f"Epoch [{epoch+1}/{config.EPOCHS}]")

    for noisy, clean in pbar:
        noisy, clean = noisy.to(device), clean.to(device)

        optimizer.zero_grad()
        pred = model(noisy)
        loss = criterion(pred, clean)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    print(f"Epoch {epoch+1} Avg Loss: {running_loss/len(loader):.4f}")
