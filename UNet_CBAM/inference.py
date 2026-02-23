import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as T

from Unet import unet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_ROOT = "test_images"        # folder containing noisy test images
SAVE_DIR = "inference_outputs"
MODEL_PATH = "best_model.pth"    # trained checkpoint

os.makedirs(SAVE_DIR, exist_ok=True)

model = unet(in_channel=1, num_classes=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


transform = T.Compose([
T.ToTensor()
])

def save_result(noisy, pred, path):
    noisy = noisy.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.imshow(noisy, cmap="gray")
    plt.title("Noisy")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(pred, cmap="gray")
    plt.title("Denoised")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


image_files = [
f for f in os.listdir(TEST_ROOT)
if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

with torch.no_grad():
    for img_name in tqdm(image_files):
        img_path = os.path.join(TEST_ROOT, img_name)

        img = Image.open(img_path).convert("L")
        img = transform(img).unsqueeze(0).to(DEVICE)

        with torch.autocast("cuda", torch.float16):
            pred = 1 - torch.sigmoid(model(img))

        pred = pred.clamp(0,1)

        save_path = os.path.join(SAVE_DIR, img_name)
        save_result(img[0], pred[0], save_path)

print("Inference completed. Outputs saved in:", SAVE_DIR)
