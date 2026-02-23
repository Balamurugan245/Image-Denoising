import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from config import Config
from model_unetpp import UNetPP
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = UNetPP().to(DEVICE)

TEST_NOISY_DIR = "/kaggle/input/testdata2-0/To_test" 
OUTPUT_DIR = "/kaggle/working/test_denoised"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model.eval()

test_tf = T.Compose([
    T.Resize((Config.img_size, Config.img_size)),
    T.ToTensor()
])

def save_pair(noisy, pred, path):
    noisy = noisy.cpu().squeeze()
    pred  = pred.cpu().squeeze()

    fig, ax = plt.subplots(1,2, figsize=(8,4))
    ax[0].imshow(noisy, cmap="gray")
    ax[0].set_title("Noisy")

    ax[1].imshow(pred, cmap="gray")
    ax[1].set_title("Prediction")

    for a in ax:
        a.axis("off")

    plt.savefig(path, dpi=200)
    plt.close()


with torch.no_grad():
    for img_name in sorted(os.listdir(TEST_NOISY_DIR)):
        img = Image.open(os.path.join(TEST_NOISY_DIR, img_name)).convert("L")
        noisy = test_tf(img).unsqueeze(0).to(DEVICE)

        pred = model(noisy)
        pred = torch.sigmoid(pred)
        pred = torch.clamp(pred, 0, 1)
        pred = 1 - pred

        save_pair(noisy, pred,
                  os.path.join(OUTPUT_DIR, img_name))

print("Saved denoised pair images to:", OUTPUT_DIR)