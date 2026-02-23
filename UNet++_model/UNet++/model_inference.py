import torch
from model_unetpp import UNetPP
from config import Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():

    model = UNetPP().to(DEVICE)

    model.load_state_dict(
        torch.load(Config.model_path, map_location=DEVICE)
    )

    model.eval()

    return model
