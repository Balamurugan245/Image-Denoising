import os

class Config:

    if os.path.exists("/kaggle/input"):
        noisy_dir = "/kaggle/input/inputdata-2k/2kdata/Dataset-1k/New_Data100/Noisy"
        clean_dir = "/kaggle/input/inputdata-2k/2kdata/Dataset-1k/New_Data100/Clean"
    else:
        noisy_dir = r"C:\Users\ADMIN\Downloads\2kdata\Dataset-1k\New_Data100\Noisy"
        clean_dir = r"C:\Users\ADMIN\Downloads\2kdata\Dataset-1k\New_Data100\Clean"

    img_size = 512
    channels = 1
    batch_size = 2
    epochs = 30
    lr = 1e-4
    checkpoint_dir = "checkpoints"
    prediction_dir = "predictions"
    val_ratio = 0.2
    num_workers = 0
    pin_memory = True
    model_path = "checkpoints/best_model.pth"
    test_dir = "test_noisy"
    output_dir = "test_output"