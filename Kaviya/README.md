# Image Denoising using UNet

This project implements an **image denoising model** using a **UNet architecture** in PyTorch.  
The model is trained on paired **noisy and clean images** to remove noise while preserving structural details.

- Architecture: UNet  
- Loss: L1 Loss + SSIM Loss  
- Framework: PyTorch  

## Structure
- `entrypoint/` – Training and inference scripts  
- `src/` – Model, dataset, loss, and pipelines  
- `config/` – Configuration file  
- `data/` – Datasets  

## 📓 Google Colab Notebook

Run this project interactively on Google Colab:
🔗(https://colab.research.google.com/drive/1YKwdbUpzP1u0mHi-0LS2BNs9tFMA6OD8)

## Train & Inference
```bash
python entrypoint/train.py
python entrypoint/inference.py


Author: Kaviya
