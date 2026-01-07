import torch
from torch.utils.data import DataLoader
from config import Config
from dataset import DenoisingDataset
from Unet import UNet
from train_model import train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

train_ds = DenoisingDataset(
    f"{Config.data_root}/Noisy",
    f"{Config.data_root}/Clean",
    split="train"
)
val_ds = DenoisingDataset(
    f"{Config.data_root}/Noisy",
    f"{Config.data_root}/Clean",
    split="val"
)

train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)

model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=Config.lr,
    weight_decay=Config.weight_decay
)

trainer = train(train_loader, val_loader, optimizer, DEVICE)
trainer.start(model, Config.epochs, Config.checkpoint_dir)



