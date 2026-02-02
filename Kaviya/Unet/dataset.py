import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from Config import Config
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_geometric_aug(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # replace ShiftScaleRotate (warning) with Affine
        A.Affine(
            translate_percent=0.05,
            scale=(0.95, 1.05),
            rotate=(-10, 10),
            p=0.5
        ),
    ])

def get_noisy_only_aug():
    return A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(var_limit=(10.0, 40.0), p=1.0),
        ], p=0.7),
    ])

def get_to_tensor():
    return A.Compose([
        A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
        ToTensorV2()
    ])

class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, split="train", val_ratio=0.2, image_size=512):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.split = split

        noisy_images = sorted(os.listdir(noisy_dir))
        clean_images = sorted(os.listdir(clean_dir))
        assert len(noisy_images) == len(clean_images)

        indices = list(range(len(noisy_images)))
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=42, shuffle=True
        )

        self.indices = train_idx if split == "train" else val_idx
        self.noisy_images = noisy_images
        self.clean_images = clean_images

        self.geo_aug = get_geometric_aug(image_size)
        self.noisy_aug = get_noisy_only_aug() if split == "train" else None
        self.to_tensor = get_to_tensor()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        noisy_path = os.path.join(self.noisy_dir, self.noisy_images[real_idx])
        clean_path = os.path.join(self.clean_dir, self.clean_images[real_idx])

        noisy = np.array(Image.open(noisy_path).convert("L"))
        clean = np.array(Image.open(clean_path).convert("L"))
        out = self.geo_aug(image=noisy, mask=clean)
        noisy, clean = out["image"], out["mask"]

        if self.noisy_aug is not None:
            noisy = self.noisy_aug(image=noisy)["image"]
        noisy = self.to_tensor(image=noisy)["image"]   # [1,H,W]
        clean = self.to_tensor(image=clean)["image"]   # [1,H,W]

        return noisy, clean
