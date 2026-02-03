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

        # Replace ShiftScaleRotate with Affine (new albumentations versions)
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.95, 1.05),
            rotate=(-10, 10),
            mode=cv2.BORDER_CONSTANT,
            cval=1.0,        
            cval_mask=1.0,  
            p=0.5
        ),

        # only affects image (not mask)
        A.RandomBrightnessContrast(p=0.3),

        ToTensorV2()
    ])


def get_val_augmentation(image_size=384):
    return A.Compose([
        A.Resize(image_size, image_size),
        ToTensorV2()
    ])


class DenoisingDataset(Dataset):
    def __init__(
        self,
        noisy_dir,
        clean_dir,
        split="train",
        val_ratio=0.2,
        image_size=384,
        invert_target=True,
        seed=42
    ):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.split = split.lower()
        self.image_size = image_size
        self.invert_target = invert_target

        self.transform = (
            get_train_augmentation(image_size=image_size) if self.split == "train"
            else get_val_augmentation(image_size=image_size)
        )

        noisy_files = sorted(os.listdir(noisy_dir))
        clean_files = set(os.listdir(clean_dir))

        self.files = [f for f in noisy_files if f in clean_files]
        if len(self.files) == 0:
            raise RuntimeError("No matching filenames found between Noisy/ and Clean/ folders.")

        indices = list(range(len(self.files)))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True
        )
        self.indices = train_idx if self.split == "train" else val_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        fname = self.files[real_idx]

        noisy_path = os.path.join(self.noisy_dir, fname)
        clean_path = os.path.join(self.clean_dir, fname)

        noisy = Image.open(noisy_path).convert("L")
        clean = Image.open(clean_path).convert("L")

        noisy = np.array(noisy, dtype=np.float32) / 255.0
        clean = np.array(clean, dtype=np.float32) / 255.0

        if self.invert_target:
            clean = 1.0 - clean

        augmented = self.transform(image=noisy, mask=clean)

        noisy_t = augmented["image"].float()
        clean_t = augmented["mask"].float()

        if noisy_t.ndim == 2:
            noisy_t = noisy_t.unsqueeze(0)
        if clean_t.ndim == 2:
            clean_t = clean_t.unsqueeze(0)

        return noisy_t, clean_t
