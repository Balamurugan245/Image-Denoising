import yaml
import torch

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def get_device(cfg):
    return "cuda" if torch.cuda.is_available() and cfg["device"] == "auto" else "cpu"
