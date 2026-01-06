import torch
from config import Config
from dataset import get_dataloaders
from Unet import UNet
from train_model import Trainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    train_loader, val_loader = get_dataloaders()

    model = UNet().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    loss_fn = Config.loss_fn

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=DEVICE
    )

    trainer.train()

if __name__ == "__main__":
    main()

