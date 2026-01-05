# UNet-CBAM Image Denoising

This project implements an image denoising model using a UNet architecture
enhanced with CBAM (Channel and Spatial Attention).

---

## Project Structure
```
Bala/
│
├── src/
│ ├── train.py # Main training script
│ ├── dataset.py # Dataset loader
│ ├── model.py # UNet + CBAM model
│ ├── loss.py # IoU loss
│ └── visualize.py # Visualization utilities
│
├── config.py # Hyperparameters and paths
├── UNet_training.ipynb #colab file
├── requirements.txt
└── README.md
```
---

## Dataset Structure

The dataset should be arranged as:
```
Dataset-1k/New_Data100/
├── Noisy/
│ ├── image1.png
│ └── image2.png
└── Clean/
├── image1.png
└── image2.png
```
---

## How to Run the Code

### 1. Install Dependencies

pip install -r requirements.txt

### 2. Configure Paths and Hyperparameters

Edit `config.py` to set:
- Dataset path
- Batch size
- Epochs
- Learning rate

---

### 3. Train the Model

python src/train.py

Training logs will be printed in the terminal.

---

## Model Description

- UNet encoder–decoder architecture
- CBAM attention blocks for feature refinement
- IoU-based loss for structural consistency

---

## Visualization

Predictions can be visualized using functions provided in `visualize.py`.
The visualization shows:
- Noisy input image
- Predicted denoised output
- Clean ground truth image

---
