# UNet-CBAM Image Denoising

This project implements an image denoising model using a UNet architecture
enhanced with CBAM (Channel and Spatial Attention).

The code is written in a clean, modular format so that it can run on
any system (local machine, server, Kaggle, or Colab).

---

## Project Structure

unet-cbam-denoising/
│
├── src/
│ ├── train.py # Training script
│ ├── dataset.py # Dataset loader
│ ├── model.py # UNet + CBAM model
│ ├── loss.py # IoU loss
│ └── visualize.py # Visualization utilities
│
├── config.py # Hyperparameters and paths
├── notebooks/
│ └── UNet_training.ipynb
│
├── requirements.txt
└── README.md

yaml
Copy code

---

## Dataset Structure

The dataset should be arranged as:

Dataset-1k/New_Data100/
├── Noisy/
│ ├── image1.png
│ └── image2.png
└── Clean/
├── image1.png
└── image2.png

yaml
Copy code

Noisy and clean images must have the same filenames.

---

## Training

Install dependencies:

```bash
pip install -r requirements.txt
Run training:

bash
Copy code
python src/train.py
All training parameters are defined in config.py.

Model Description
UNet encoder–decoder architecture

CBAM attention blocks for feature refinement

IoU-based loss for structural consistency

Visualization
Predictions can be visualized using the functions provided in
visualize.py, which show noisy input, predicted output, and clean ground truth.

Author
Academic deep learning project

yaml
Copy code

---

If you want, next I can give you:
- `requirements.txt` (single copy-paste)
- `config.py` (single copy-paste)
- a **sir-impressing** short README (even more minimal)

Just tell me 👍






