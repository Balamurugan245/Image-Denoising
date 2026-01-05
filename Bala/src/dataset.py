# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class DenoiseDataset(Dataset):
    def __init__(self, root, img_size=512):
        self.noisy_dir = os.path.join(root, "Noisy")
        self.clean_dir = os.path.join(root, "Clean")

        noisy_files = os.listdir(self.noisy_dir)
        clean_files = os.listdir(self.clean_dir)

        noisy_map = {os.path.splitext(f)[0]: f for f in noisy_files}
        clean_map = {os.path.splitext(f)[0]: f for f in clean_files}

        self.keys = sorted(noisy_map.keys() & clean_map.keys())
        self.noisy_map = noisy_map
        self.clean_map = clean_map

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

        print("Total matched image pairs:", len(self.keys))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        noisy = Image.open(
            os.path.join(self.noisy_dir, self.noisy_map[key])
        ).convert("RGB")

        clean = Image.open(
            os.path.join(self.clean_dir, self.clean_map[key])
        ).convert("RGB")

        return self.transform(noisy), self.transform(clean)
