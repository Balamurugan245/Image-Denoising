import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import UNet
from src.dataset import DrawingDenoiseDataset
from src.loss_function import SSIMLoss
from src.train_pipeline import train
from src.utils import load_config, get_device

cfg = load_config("config/config.yaml")
device = get_device(cfg)

dataset = DrawingDenoiseDataset(
    cfg["paths"]["noisy_train"],
    cfg["paths"]["clean_train"],
    cfg["training"]["img_size"]
)

loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)

model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

train(model, loader, optimizer, nn.L1Loss(), SSIMLoss(), cfg, device)

torch.save(model.state_dict(), cfg["paths"]["save_model"])
