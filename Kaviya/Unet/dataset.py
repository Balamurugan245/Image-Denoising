import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_augmentation(image_size=384):
    return A.Compose([
        A.Resize(image_size, image_size),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            value=255,       # input background white
            mask_value=255,  # target background white (before invert)
            p=0.5
        ),

        A.RandomBrightnessContrast(p=0.3),
        ToTensorV2()
    ])

def get_val_augmentation(image_size=384):
    return A.Compose([
        A.Resize(image_size, image_size),
        ToTensorV2()
    ])

class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, split="train", val_ratio=0.2, image_size=384, invert_target=True):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.split = split.lower()
        self.image_size = image_size
        self.invert_target = invert_target

        self.transform = (
            get_train_augmentation(image_size=image_size) if self.split == "train"
            else get_val_augmentation(image_size=image_size)
        )

        self.noisy_images = sorted(os.listdir(noisy_dir))
        self.clean_images = sorted(os.listdir(clean_dir))
        assert len(self.noisy_images) == len(self.clean_images), "Noisy/Clean count mismatch"

        indices = list(range(len(self.noisy_images)))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=42,
            shuffle=True
        )

        self.indices = train_idx if self.split == "train" else val_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        noisy_path = os.path.join(self.noisy_dir, self.noisy_images[real_idx])
        clean_path = os.path.join(self.clean_dir, self.clean_images[real_idx])

        noisy = Image.open(noisy_path).convert("L")
        clean = Image.open(clean_path).convert("L")

        noisy = np.array(noisy, dtype=np.float32)  # [H,W] 0..255
        clean = np.array(clean, dtype=np.float32)

        noisy = noisy / 255.0
        clean = clean / 255.0

        if self.invert_target:
            clean = 1.0 - clean

        augmented = self.transform(image=noisy, mask=clean)
        noisy_t = augmented["image"].float()  # [1,H,W] float32
        clean_t = augmented["mask"].float()   # [H,W] or [1,H,W] depending on ToTensorV2 version

        if clean_t.ndim == 2:
            clean_t = clean_t.unsqueeze(0)

        if noisy_t.ndim == 2:
            noisy_t = noisy_t.unsqueeze(0)

        return noisy_t, clean_t
