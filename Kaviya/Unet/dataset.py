import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_augmentation():
    return A.Compose([
        A.Resize(Config.image_size, Config.image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=10,
            p=0.5
        ),
        A.RandomBrightnessContrast(p=0.3),
        ToTensorV2()
    ])

def get_val_augmentation():
    return A.Compose([
        A.Resize(Config.image_size, Config.image_size),
        ToTensorV2()
    ])

class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, split="train", val_ratio=0.2):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.split = split

        self.transform = (
            get_train_augmentation() if split == "train"
            else get_val_augmentation()
        )

        noisy_images = sorted(os.listdir(noisy_dir))
        clean_images = sorted(os.listdir(clean_dir))
        assert len(noisy_images) == len(clean_images)

        indices = list(range(len(noisy_images)))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=42,
            shuffle=True
        )

        self.indices = train_idx if split == "train" else val_idx
        self.noisy_images = noisy_images
        self.clean_images = clean_images

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        noisy = Image.open(
            os.path.join(self.noisy_dir, self.noisy_images[real_idx])
        ).convert("L")

        clean = Image.open(
            os.path.join(self.clean_dir, self.clean_images[real_idx])
        ).convert("L")

        noisy = np.array(noisy)
        clean = np.array(clean)
        augmented = self.transform(image=noisy, mask=clean)
        noisy = augmented["image"] 
        clean = augmented["mask"]

        return noisy, clean



