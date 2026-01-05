import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class DenoiseDataset(Dataset):
    def __init__(self, root, img_size=512):
        self.noisy_dir = os.path.join(root, "Noisy")
        self.clean_dir = os.path.join(root, "Clean")

        self.files = sorted(os.listdir(self.noisy_dir))

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = Image.open(
            os.path.join(self.noisy_dir, self.files[idx])
        ).convert("RGB")

        y = Image.open(
            os.path.join(self.clean_dir, self.files[idx])
        ).convert("RGB")

        return self.transform(x), self.transform(y)
