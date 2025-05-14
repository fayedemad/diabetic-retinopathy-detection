import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import os
from ai_model.config import TRAIN_CONFIG, SCHEDULER_CONFIG, MODEL_DIR
from ai_model.models.model import get_model

class Trainer:
    def __init__(self, train_loader, val_loader, config=TRAIN_CONFIG):
        self.config = config
        self.device = torch.device(config["device"])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = get_model(device=self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=SCHEDULER_CONFIG["T_0"],
            T_mult=SCHEDULER_CONFIG["T_mult"],
            eta_min=SCHEDULER_CONFIG["eta_min"]
        )
        
        self.best_val_acc = 0
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': total_loss/total,
                'acc': 100.*correct/total
            })
        
        return total_loss/len(self.train_loader), 100.*correct/total
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss/len(self.val_loader), 100.*correct/total
    
    def save_checkpoint(self, epoch, val_acc):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc
        }
        
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            torch.save(checkpoint, os.path.join(MODEL_DIR, 'best_model.pth'))
        
        torch.save(checkpoint, os.path.join(MODEL_DIR, 'latest_model.pth'))
    
    def train(self):
        """Train the model for specified number of epochs."""
        for epoch in range(self.config["num_epochs"]):
            print(f'\nEpoch {epoch+1}/{self.config["num_epochs"]}')
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            self.save_checkpoint(epoch, val_acc)