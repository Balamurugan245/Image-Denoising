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

## Train
```bash
python entrypoint/train.py


## Inference
```bash
python entrypoint/inference.py


Author: Kaviya
