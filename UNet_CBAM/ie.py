import psutil
import torch as t
import gc
import numpy as np
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
print("CUDA Available:", t.cuda.is_available())
print("Using Device:", DEVICE)

def resources_cpu(memory, memory_used=None):
    total_memory = memory.total
    available_memory = memory.available
    used_memory = total_memory - available_memory

    print("\nRAM STATUS:")
    print(f"    Used memory: {used_memory / (1024**3):.2f} GB")
    print(f"    Available memory: {available_memory / (1024**3):.2f} GB")
    print(f"    Memory usage percentage: {memory.percent}%")

    if memory_used:
        print(f"    Additional RAM used: {(used_memory-memory_used)/(1024**3):.2f} GB")

    print("----"*20)
    return used_memory


def resources_gpu(memory, memory_used=None):
    used_memory = resources_cpu(memory, memory_used)

    print("GPU STATUS:")
    if t.cuda.is_available():
        print(f"    GPU Memory Used: {t.cuda.memory_allocated()/(1024*1024):.2f} MB")

    print("----"*20)
    return used_memory

memory = psutil.virtual_memory()
memory_used = resources_cpu(memory)

MODEL_PATH = "best_model.pth"
INPUT_IMAGE = "test/noisy.png"
OUTPUT_IMAGE = "output/denoised.png"

os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)

print("\nBEFORE LOADING MODEL")
memory = psutil.virtual_memory()
memory_used = resources_gpu(memory, memory_used)

model = t.load(MODEL_PATH, map_location=DEVICE)
model.to(DEVICE)
model.eval()

print("AFTER LOADING MODEL")
memory = psutil.virtual_memory()
memory_used = resources_gpu(memory, memory_used)

transform_pipe = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

print("\nLOADING INPUT IMAGE...")

img = Image.open(INPUT_IMAGE).convert("L")  
img_tensor = transform_pipe(img)
img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

memory = psutil.virtual_memory()
memory_used = resources_gpu(memory, memory_used)

print("\nRUNNING PREDICTION...")

with t.inference_mode():
    pred = model(img_tensor)

memory = psutil.virtual_memory()
memory_used = resources_gpu(memory, memory_used)

output_img = pred.squeeze().cpu().numpy()
output_img = (output_img * 255).clip(0, 255).astype(np.uint8)

Image.fromarray(output_img).save(OUTPUT_IMAGE)

print("\n DENOISED IMAGE SAVED AT:", OUTPUT_IMAGE)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Noisy Input")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Denoised Output")
plt.imshow(output_img, cmap="gray")
plt.axis("off")

plt.show()

del img_tensor, pred
gc.collect()
t.cuda.empty_cache()

print("\nAFTER MEMORY CLEANUP")
memory = psutil.virtual_memory()
memory_used = resources_gpu(memory, memory_used)