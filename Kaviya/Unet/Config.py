class Config:
    image_size = 512
    batch_size = 2
    val_ratio = 0.2

    epochs = 30
    lr = 1e-4
    weight_decay = 1e-5

    ssim_weight = 0.5

    data_root = "/kaggle/input/input-data2k/2kdata/Dataset-1k/New_Data100"
    checkpoint_dir = "checkpoints"
    prediction_dir = "predictions"
