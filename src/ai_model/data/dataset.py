from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from ai_model.utils.transforms import get_transforms
from ai_model.config import DATASET_CONFIG, TRAIN_CONFIG

class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, image_paths, labels, phase='train'):
        self.image_paths = image_paths
        self.labels = labels
        self.phase = phase
        self.transforms = get_transforms(phase)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        
        return image, label

def get_image_paths_and_labels(data_dir):
    """Get all image paths and their corresponding labels."""
    image_paths = []
    labels = []
    
    for class_idx in range(5):  # 5 classes for DR stages
        class_dir = os.path.join(data_dir, f'class_{class_idx}')
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir):
                if any(img_name.lower().endswith(ext) for ext in DATASET_CONFIG["image_extensions"]):
                    image_paths.append(os.path.join(class_dir, img_name))
                    labels.append(class_idx)
    
    return np.array(image_paths), np.array(labels)

def create_dataloaders(data_dir, config=TRAIN_CONFIG):
    """Create train and validation dataloaders."""
    # Get all image paths and labels
    image_paths, labels = get_image_paths_and_labels(data_dir)
    
    # Shuffle the data
    indices = np.random.permutation(len(image_paths))
    image_paths = image_paths[indices]
    labels = labels[indices]
    
    # Split into train and validation
    split_idx = int(len(image_paths) * DATASET_CONFIG["train_ratio"])
    train_paths = image_paths[:split_idx]
    train_labels = labels[:split_idx]
    val_paths = image_paths[split_idx:]
    val_labels = labels[split_idx:]
    
    # Create datasets
    train_dataset = DiabeticRetinopathyDataset(train_paths, train_labels, phase='train')
    val_dataset = DiabeticRetinopathyDataset(val_paths, val_labels, phase='val')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )
    
    return train_loader, val_loader 