import yaml
import torch
import random
import numpy as np

def load_config(path="config/local.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
