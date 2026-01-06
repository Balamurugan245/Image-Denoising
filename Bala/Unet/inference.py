import torch
from PIL import Image
import torchvision.transforms as T

def infer(model, image_path, device):
    t = T.Compose([T.Resize((512,512)), T.ToTensor()])
    img = t(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img)
    return pred
