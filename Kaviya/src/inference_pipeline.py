import os
import torch
import torchvision.utils as vutils

def inference(model, loader, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for noisy, name in loader:
            noisy = noisy.to(device)
            pred = model(noisy).clamp(0, 1)
            vutils.save_image(pred, os.path.join(output_dir, name[0]))
