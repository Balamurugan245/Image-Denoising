import sys, os
sys.path.append(os.path.abspath(".."))

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from src.model import UNet
from src.noisy_dataset import NoisyOnlyDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)
model.load_state_dict(torch.load("unet_denoiser.pth", map_location=device))
model.eval()

dataset = NoisyOnlyDataset("data/01-raw/noisy")
loader = DataLoader(dataset, batch_size=1)

os.makedirs("data/04-predictions", exist_ok=True)

with torch.no_grad():
    for noisy, name in loader:
        noisy = noisy.to(device)
        pred = model(noisy)
        vutils.save_image(pred, f"data/04-predictions/{name[0]}")
