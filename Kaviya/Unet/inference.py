import torch
from unet import UNet
from torchvision.utils import save_image


def run_inference(model_path, input_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor.to(device))

    save_image(output, "denoised.png")
