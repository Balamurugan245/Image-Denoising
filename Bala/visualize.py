# visualize.py
import torch
import random
import matplotlib.pyplot as plt

def visualize_predictions(model, dataset, device, num_samples=10):
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)

    plt.figure(figsize=(12, num_samples * 3))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            noisy, clean = dataset[idx]
            pred = model(noisy.unsqueeze(0).to(device))

            noisy = noisy.permute(1,2,0).cpu()
            clean = clean.permute(1,2,0).cpu()
            pred  = pred.squeeze(0).permute(1,2,0).cpu()

            plt.subplot(num_samples, 3, i*3+1)
            plt.imshow(noisy); plt.title("Noisy"); plt.axis("off")

            plt.subplot(num_samples, 3, i*3+2)
            plt.imshow(pred); plt.title("Predicted"); plt.axis("off")

            plt.subplot(num_samples, 3, i*3+3)
            plt.imshow(clean); plt.title("Clean"); plt.axis("off")

    plt.tight_layout()
    plt.show()
