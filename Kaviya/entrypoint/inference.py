import torch
from torch.utils.data import DataLoader

from src.model import UNet
from src.dataset import NoisyOnlyDataset
from src.inference_pipeline import inference
from src.utils import load_config, get_device

cfg = load_config("config/config.yaml")
device = get_device(cfg)

model = UNet().to(device)
model.load_state_dict(torch.load(cfg["paths"]["save_model"], map_location=device))

dataset = NoisyOnlyDataset(cfg["paths"]["test_dir"])
loader = DataLoader(dataset, batch_size=1)

inference(model, loader, cfg["paths"]["output_dir"], device)
