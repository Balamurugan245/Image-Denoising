# inference.py
import torch
import torchvision.transforms as T
from PIL import Image
from models.Unet import UNetCBAM

def load_model(weight_path, device):
    model = UNetCBAM().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model

def run_inference(image_path, model, device):
    t = T.Compose([T.Resize((512,512)), T.ToTensor()])
    img = t(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
    return output
