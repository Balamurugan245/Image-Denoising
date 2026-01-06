import torch
import torch.nn as nn
from tqdm import tqdm

class IoULoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.clamp(pred, 1e-6, 1-1e-6)
        target = (target > 0.1).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return 1 - (intersection + 1e-6) / (union + 1e-6)


def train(model, train_loader, val_loader, device, epochs=50):
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = IoULoss()
    scaler = torch.amp.GradScaler("cuda")

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for noisy, clean in tqdm(train_loader):
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                pred = model(noisy)
                loss = criterion(pred, clean)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train {total_loss:.4f}, Val {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")


def validate(model, loader, criterion, device):
    model.eval()
    loss = 0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            loss += criterion(model(noisy), clean).item()
    return loss / len(loader)
