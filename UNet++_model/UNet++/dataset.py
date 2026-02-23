import os
import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class DenoisingDataset(Dataset):
    def __init__(
        self,
        noisy_dir,
        clean_dir,
        split="train",
        val_ratio=0.2,
        image_size=512,
        invert_target=False,
        channels=1,
        augment=True
    ):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.split = split.lower()
        self.invert_target = invert_target
        self.channels = channels
        self.image_size = image_size
        self.augment = augment if split == "train" else False

        noisy_images = sorted(os.listdir(noisy_dir))
        clean_images = sorted(os.listdir(clean_dir))

        noisy_map = {os.path.splitext(f)[0]: f for f in noisy_images}
        clean_map = {os.path.splitext(f)[0]: f for f in clean_images}
        common_keys = sorted(list(set(noisy_map.keys()) & set(clean_map.keys())))

        if len(common_keys) == 0:
            raise ValueError("No matching noisy-clean pairs found!")

        self.noisy_images = [noisy_map[k] for k in common_keys]
        self.clean_images = [clean_map[k] for k in common_keys]

        print(f"Total matched images: {len(self.noisy_images)}")

        indices = list(range(len(self.noisy_images)))
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=42, shuffle=True
        )

        self.indices = train_idx if self.split == "train" else val_idx

    def __len__(self):
        return len(self.indices)

    def apply_augmentation(self, noisy, clean):
        if random.random() > 0.5:
            noisy = noisy.transpose(Image.FLIP_LEFT_RIGHT)
            clean = clean.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() > 0.5:
            noisy = noisy.transpose(Image.FLIP_TOP_BOTTOM)
            clean = clean.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            noisy = noisy.rotate(angle)
            clean = clean.rotate(angle)

        return noisy, clean

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        noisy_path = os.path.join(self.noisy_dir, self.noisy_images[real_idx])
        clean_path = os.path.join(self.clean_dir, self.clean_images[real_idx])

        try:
            if self.channels == 3:
                noisy = Image.open(noisy_path).convert("RGB")
                clean = Image.open(clean_path).convert("RGB")
            else:
                noisy = Image.open(noisy_path).convert("L")
                clean = Image.open(clean_path).convert("L")
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        noisy = noisy.resize((self.image_size, self.image_size), Image.BILINEAR)
        clean = clean.resize((self.image_size, self.image_size), Image.BILINEAR)

        if self.augment:
            noisy, clean = self.apply_augmentation(noisy, clean)

        noisy = np.array(noisy, dtype=np.float32) / 255.0
        clean = np.array(clean, dtype=np.float32) / 255.0

        if self.invert_target:
            clean = 1.0 - clean

        if self.channels == 1:
            noisy = np.expand_dims(noisy, axis=0)
            clean = np.expand_dims(clean, axis=0)
        else:
            noisy = np.transpose(noisy, (2, 0, 1))
            clean = np.transpose(clean, (2, 0, 1))

        noisy_t = torch.tensor(noisy, dtype=torch.float32)
        clean_t = torch.tensor(clean, dtype=torch.float32)

        return noisy_t, clean_t