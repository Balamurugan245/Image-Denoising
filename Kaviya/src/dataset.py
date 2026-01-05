import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class DrawingDenoiseDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, img_size=224):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir

        noisy_map = {os.path.splitext(f)[0]: f for f in os.listdir(noisy_dir)}
        clean_map = {os.path.splitext(f)[0]: f for f in os.listdir(clean_dir)}

        self.keys = sorted(noisy_map.keys() & clean_map.keys())
        self.noisy_map = noisy_map
        self.clean_map = clean_map

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        noisy = Image.open(os.path.join(self.noisy_dir, self.noisy_map[key])).convert("RGB")
        clean = Image.open(os.path.join(self.clean_dir, self.clean_map[key])).convert("RGB")
        return self.transform(noisy), self.transform(clean)


class NoisyOnlyDataset(Dataset):
    def __init__(self, noisy_dir, img_size=224):
        self.files = sorted(os.listdir(noisy_dir))
        self.noisy_dir = noisy_dir
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.noisy_dir, name)).convert("RGB")
        return self.transform(img), name
