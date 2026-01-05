import os
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from src.dataset import NoisyOnlyDataset
from src.model import UNet
from src.utils import show_samples

device = "cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("unet_denoiser.pth", map_location=device))
model.eval()

test_dataset = NoisyOnlyDataset("data/test/noisy")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
os.makedirs("predicted_outputs", exist_ok=True)

with torch.no_grad():
    for noisy, name in test_loader:
        noisy = noisy.to(device)
        pred = model(noisy).clamp(0,1)
        save_path = os.path.join("predicted_outputs", name[0])
        vutils.save_image(pred, save_path)

show_samples(model, test_dataset, device, num=5)
