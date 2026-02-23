from train_model import train
from inference import inference


def run(mode):
    if mode == "train":
        train()

    elif mode == "inference":
        inference()

    else:
        print("Invalid mode")
