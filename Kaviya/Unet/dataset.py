import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, split="train", val_ratio=0.2):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir

        noisy_images = sorted(os.listdir(noisy_dir))
        clean_images = sorted(os.listdir(clean_dir))

        assert len(noisy_images) == len(clean_images), \
            "Noisy and Clean folders must have same number of images"

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
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        noisy_path = os.path.join(self.noisy_dir, self.noisy_images[real_idx])
        clean_path = os.path.join(self.clean_dir, self.clean_images[real_idx])

        noisy = Image.open(noisy_path).convert("L")
        clean = Image.open(clean_path).convert("L")

        return self.transform(noisy), self.transform(clean)
