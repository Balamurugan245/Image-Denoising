import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DenoiseDataset(Dataset):
    def __init__(self, root, transform):
        self.noisy_dir = os.path.join(root, "Noisy")
        self.clean_dir = os.path.join(root, "Clean")

        noisy_map = {os.path.splitext(f)[0]: f for f in os.listdir(self.noisy_dir)}
        clean_map = {os.path.splitext(f)[0]: f for f in os.listdir(self.clean_dir)}

        self.keys = sorted(noisy_map.keys() & clean_map.keys())
        self.noisy_map = noisy_map
        self.clean_map = clean_map
        self.transform = transform

        print("Total matched image pairs:", len(self.keys))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
    
        noisy = np.array(Image.open(
            os.path.join(self.noisy_dir, self.noisy_map[key])
        ).convert("L"))
    
        clean = np.array(Image.open(
            os.path.join(self.clean_dir, self.clean_map[key])
        ).convert("L"))
    
        augmented = self.transform(image=noisy, mask=clean)
    
        noisy = augmented["image"].float() / 255.0
        clean = augmented["mask"].float() / 255.0
        
        return noisy, clean

