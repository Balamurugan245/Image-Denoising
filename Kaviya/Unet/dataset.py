import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DenoiseDataset(Dataset):
    def __init__(self, root, img_size=384, invert_target=True, augment=False):
        self.noisy_dir = os.path.join(root, "Noisy")
        self.clean_dir = os.path.join(root, "Clean")
        self.files = sorted(os.listdir(self.noisy_dir))

        self.invert_target = invert_target

        tf_list = [A.Resize(img_size, img_size)]

        if augment:
            tf_list += [
                A.ShiftScaleRotate(
                    shift_limit=0.02, scale_limit=0.05, rotate_limit=5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=1.0,       
                    mask_value=1.0, 
                    p=0.7
                ),
            ]

        tf_list += [ToTensorV2()]
        self.tf = A.Compose(tf_list)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        noisy_path = os.path.join(self.noisy_dir, fname)
        clean_path = os.path.join(self.clean_dir, fname)

        x = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
        y = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
        if x is None or y is None:
            raise FileNotFoundError(f"Bad image read: {noisy_path} or {clean_path}")

        x = x.astype(np.float32) / 255.0
        y = y.astype(np.float32) / 255.0

        if self.invert_target:
            y = 1.0 - y

        out = self.tf(image=x, mask=y)
        x_t = out["image"].unsqueeze(0)  # [1,H,W]
        y_t = out["mask"].unsqueeze(0)   # [1,H,W]

        return x_t, y_t
