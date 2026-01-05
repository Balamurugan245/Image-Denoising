import matplotlib.pyplot as plt
import torch

def show_samples(model, dataset, device, num=5):
    model.eval()
    idxs = torch.randperm(len(dataset))[:num]

    plt.figure(figsize=(12, 3 * num))
    for i, idx in enumerate(idxs):
        noisy, clean = dataset[idx]
        noisy_b = noisy.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(noisy_b)

        noisy = noisy.permute(1,2,0).cpu()
        clean = clean.permute(1,2,0)
        pred = pred.squeeze().permute(1,2,0).cpu()

        plt.subplot(num, 3, i*3+1)
        plt.imshow(noisy)
        plt.title("Noisy")
        plt.axis("off")

        plt.subplot(num, 3, i*3+2)
        plt.imshow(clean)
        plt.title("Clean GT")
        plt.axis("off")

        plt.subplot(num, 3, i*3+3)
        plt.imshow(pred)
        plt.title("UNet Pred")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
