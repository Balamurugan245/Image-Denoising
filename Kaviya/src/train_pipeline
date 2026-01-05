import time
import torch
from tqdm import tqdm

def train(model, loader, optimizer, l1_loss, ssim_loss, cfg, device):
    epochs = cfg["training"]["epochs"]

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start = time.time()

        for noisy, clean in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            pred = model(noisy)

            loss = (
                cfg["loss"]["l1_weight"] * l1_loss(pred, clean)
                + cfg["loss"]["ssim_weight"] * ssim_loss(pred, clean)
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f} | Time: {time.time()-start:.2f}s")
