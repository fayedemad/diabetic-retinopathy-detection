import os
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

for dir_path in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    dir_path.mkdir(exist_ok=True)

MODEL_CONFIG = {
    "name": "efficientnet_b4", # cuz it is good for medical diagnosis, based on imagenet
    "num_classes": 5, # number of medical classes
    "pretrained": True,
    "dropout_rate": 0.3,
    "hidden_size": 512 # hidden layers size..
}

TRAIN_CONFIG = {
    "batch_size": 8, # 3lshan fsh5ly l memory xD
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "num_workers": 4,
    "pin_memory": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

SCHEDULER_CONFIG = {
    "T_0": 10,  # Restart every 10 epochs
    "T_mult": 2,  # Double the restart interval after each restart
    "eta_min": 1e-6  # Minimum learning rate
}

# data augmentation configuration (like rotating, shades etc..)
AUGMENTATION_CONFIG = {
    "train": {
        "image_size": 380,
        "scale": (0.8, 1.0),
        "horizontal_flip_prob": 0.5,
        "vertical_flip_prob": 0.5,
        "rotate_prob": 0.5,
        "brightness_contrast_prob": 0.5,
        "elastic_transform_prob": 0.3
    },
    "val": {
        "image_size": 380
    }
}

DATASET_CONFIG = {
    "train_ratio": 0.8,
    "val_ratio": 0.2,
    "image_extensions": [".jpg", ".jpeg", ".png"]
} 