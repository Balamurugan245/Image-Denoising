import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from Unet import unet
from config import Config
from dataset import DenoiseDataset, get_train_transform, get_val_transform
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CADLoss(nn.Module):
    def __init__(self, w_charb=1.6, w_edge=1.2, eps=1e-3):
        super().__init__()
        self.w_charb = w_charb
        self.w_edge = w_edge
        self.eps = eps

        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        )

    def charbonnier(self, p, t):
        return torch.mean(torch.sqrt((p - t)**2 + self.eps**2))

    def edge_loss(self, p, t):
        px = F.conv2d(p, self.sobel_x, padding=1)
        py = F.conv2d(p, self.sobel_y, padding=1)
        tx = F.conv2d(t, self.sobel_x, padding=1)
        ty = F.conv2d(t, self.sobel_y, padding=1)
        return F.l1_loss(px, tx) + F.l1_loss(py, ty)

    def forward(self, p, t):
        return self.w_charb * self.charbonnier(p, t) + self.w_edge * self.edge_loss(p, t)


train_tf = get_train_transform()
val_tf = get_val_transform()

dataset = DenoiseDataset(Config.data_root, train_tf)

train_size = int((1 - Config.VAL_SPLIT) * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
val_ds.dataset.transform = val_tf

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

model = unet(in_channel=1, num_classes=1).to(DEVICE)

criterion = CADLoss().to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=Config.LR_PATIENCE,
    factor=Config.LR_FACTOR
)

scaler = torch.amp.GradScaler("cuda")

os.makedirs(Config.checkpoint_dir, exist_ok=True)
os.makedirs(Config.prediction_dir, exist_ok=True)

best_loss = float("inf")
fixed_noisy, fixed_clean = next(iter(val_loader))
fixed_noisy = fixed_noisy.to(DEVICE).float()
fixed_clean = fixed_clean.to(DEVICE)

def to_img(x):
    x = x.detach().cpu().float()
    x = torch.clamp(x, 0, 1)
    return x.squeeze(0).numpy()

def save_triplet(noisy, pred, clean, path, epoch):
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(to_img(noisy), cmap="gray")
    plt.title("Noisy")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(to_img(pred), cmap="gray")
    plt.title("Predicted")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(to_img(clean), cmap="gray")
    plt.title("Clean")
    plt.axis("off")

    plt.suptitle(f"Epoch {epoch}", fontsize=16)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

for epoch in range(1, Config.EPOCHS + 1):
    model.train()
    train_loss_sum = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{Config.EPOCHS}]")

    for noisy, clean in pbar:

        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", torch.float16):
            pred = model(noisy)
            loss = criterion(pred, clean)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss_sum += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = train_loss_sum / len(train_loader)

    model.eval()
    val_loss_sum = 0

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)

            with torch.autocast("cuda", torch.float16):
                pred = model(noisy)
                loss = criterion(pred, clean)

            val_loss_sum += loss.item()

    val_loss = val_loss_sum / len(val_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    torch.save(
        model.state_dict(),
        os.path.join(Config.checkpoint_dir, f"epoch_{epoch}.pth")
    )

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(
            model.state_dict(),
            os.path.join(Config.checkpoint_dir, "best_model.pth")
        )

    model.eval()
    with torch.no_grad():
        pred = model(fixed_noisy)
        pred = torch.sigmoid(pred)

    epoch_dir = os.path.join(
        Config.prediction_dir,
        f"epoch_{epoch:03d}"
    )
    os.makedirs(epoch_dir, exist_ok=True)

    NUM_SAMPLES = 10
    for i in range(min(NUM_SAMPLES, fixed_noisy.size(0))):

        img_path = os.path.join(
            epoch_dir,
            f"sample_{i+1}.png"
        )

        save_triplet(
            fixed_noisy[i],
            pred[i],
            fixed_clean[i],
            img_path,
            epoch
        )

print("Training Completed")
