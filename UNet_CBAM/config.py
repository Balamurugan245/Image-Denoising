class Config:
    IMAGE_SIZE = 512
    IMAGE_SIZE_NEW = 384
    BATCH_SIZE = 4
    VAL_SPLIT = 0.2

    EPOCHS = 40
    LR = 1e-4
    WEIGHT_DECAY = 1e-4

    USE_AMP = True
    GRAD_CLIP = 1.0
    LR_PATIENCE = 5
    LR_FACTOR = 0.5

    data_root = "2kdata/partdrawing/Dataset-1k/New_Data100"
    checkpoint_dir = "checkpoints"
    prediction_dir = "predictions"