import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

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

        noisy = Image.open(
            os.path.join(self.noisy_dir, self.noisy_map[key])
        ).convert("RGB")

        clean = Image.open(
            os.path.join(self.clean_dir, self.clean_map[key])
        ).convert("RGB")

        return self.transform(noisy), self.transform(clean)


def get_transforms(train=True):
    if train:
        return T.Compose([
            T.Resize((512, 512)),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.ColorJitter(0.2, 0.2),
            T.ToTensor()
        ])
    else:
        return T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])
