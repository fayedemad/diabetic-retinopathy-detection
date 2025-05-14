from ai_model.data.dataset import create_dataloaders
from ai_model.trainer import Trainer
from ai_model.config import DATA_DIR, TRAIN_CONFIG

def main():
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        data_dir=DATA_DIR,
        config=TRAIN_CONFIG
    )
    
    # Initialize trainer
    trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        config=TRAIN_CONFIG
    )
    
    # Train the model
    trainer.train()
    
    print("Training completed!")

if __name__ == "__main__":
    main() 