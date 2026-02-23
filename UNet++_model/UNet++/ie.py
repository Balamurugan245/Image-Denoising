import matplotlib.pyplot as plt


def save_triplet(noisy, clean, pred, path):

    noisy = noisy.cpu().squeeze()
    clean = clean.cpu().squeeze()
    pred = pred.cpu().squeeze()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(noisy, cmap="gray")
    ax[0].set_title("Noisy")

    ax[1].imshow(clean, cmap="gray")
    ax[1].set_title("Clean")

    ax[2].imshow(pred, cmap="gray")
    ax[2].set_title("Prediction")

    for a in ax:
        a.axis("off")

    plt.savefig(path)

    plt.close()
