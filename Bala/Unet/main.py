import torch
from torch.utils.data import DataLoader, random_split
from dataset import DenoiseDataset, get_transforms
from models.Unet import UNetCBAM
from train_model import train

DATA_ROOT = "Dataset-1k/New_Data100"
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = DenoiseDataset(DATA_ROOT, get_transforms(train=True))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
val_ds.dataset.transform = get_transforms(train=False)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

model = UNetCBAM().to(device)

train(model, train_loader, val_loader, device)
