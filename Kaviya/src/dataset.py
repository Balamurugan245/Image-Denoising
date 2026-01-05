from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import os

class DrawingDenoiseDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, img_size=224):
        self.files = sorted(os.listdir(noisy_dir))
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        noisy = Image.open(os.path.join(self.noisy_dir, name)).convert("RGB")
        clean = Image.open(os.path.join(self.clean_dir, name)).convert("RGB")
        return self.transform(noisy), self.transform(clean)
