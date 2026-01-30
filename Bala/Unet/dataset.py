import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config 

def get_train_transform():
    return A.Compose([
            A.Resize(Config.IMAGE_SIZE_NEW, Config.IMAGE_SIZE_NEW),
            A.ShiftScaleRotate(
                shift_limit=0.02,
                scale_limit=0.02,
                rotate_limit=2,
                border_mode=0,
                p=0.5
            ),
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            ToTensorV2()],
        additional_targets={"mask": "image"}
    )


def get_val_transform():
    return A.Compose([
            A.Resize(IMAGE_SIZE_NEW, IMAGE_SIZE_NEW),
            ToTensorV2()],
        additional_targets={"mask": "image"}
    )
    
class DenoiseDataset(Dataset):
    def __init__(self, root, transform=None):
        self.noisy_dir = os.path.join(root, "Noisy")
        self.clean_dir = os.path.join(root, "Clean")
        self.transform = transform

        noisy_map = {
            os.path.splitext(f)[0]: f
            for f in os.listdir(self.noisy_dir)
        }
        clean_map = {
            os.path.splitext(f)[0]: f
            for f in os.listdir(self.clean_dir)
        }

        self.keys = sorted(noisy_map.keys() & clean_map.keys())
        self.noisy_map = noisy_map
        self.clean_map = clean_map

        print("Total matched image pairs:", len(self.keys))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        noisy = np.array(
            Image.open(
                os.path.join(self.noisy_dir, self.noisy_map[key])
            ).convert("L"),
            dtype=np.uint8
        )
        
        clean = np.array(
            Image.open(
                os.path.join(self.clean_dir, self.clean_map[key])
            ).convert("L"),
            dtype=np.uint8
        )

        if self.transform is not None:
            augmented = self.transform(image=noisy, mask=clean)
            noisy = augmented["image"]
            clean = augmented["mask"]

        noisy = noisy.float() / 255.0
        clean = clean.float() / 255.0

        return noisy, clean


