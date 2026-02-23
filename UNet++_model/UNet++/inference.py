import torch
import os
from PIL import Image
import torchvision.transforms as T

from model_inference import load_model
from config import Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def inference():

    model = load_model()

    transform = T.Compose([
        T.Resize((Config.img_size, Config.img_size)),
        T.ToTensor()
    ])

    os.makedirs(Config.output_dir, exist_ok=True)

    model.eval()

    with torch.no_grad():

        for img_name in sorted(os.listdir(Config.test_dir)):

            img_path = os.path.join(Config.test_dir, img_name)

            img = Image.open(img_path).convert("L")

            noisy = transform(img).unsqueeze(0).to(DEVICE)

            pred = model(noisy)

            pred = torch.clamp(pred, 0, 1)

            output_path = os.path.join(Config.output_dir, img_name)

            T.ToPILImage()(pred.squeeze().cpu()).save(output_path)

    print("Inference completed")
