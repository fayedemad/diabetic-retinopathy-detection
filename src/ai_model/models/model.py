import torch
import torch.nn as nn
import torchvision.models as models
from ai_model.config import MODEL_CONFIG

class DiabeticRetinopathyModel(nn.Module):
    def __init__(self, config=MODEL_CONFIG):
        super(DiabeticRetinopathyModel, self).__init__()
        
        if config["name"] == "efficientnet_b4":
            self.base_model = models.efficientnet_b4(pretrained=config["pretrained"])
            num_features = self.base_model.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported model name: {config['name']}")
        
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=config["dropout_rate"]),
            nn.Linear(num_features, config["hidden_size"]),
            nn.ReLU(),
            nn.Dropout(p=config["dropout_rate"]/2),
            nn.Linear(config["hidden_size"], config["num_classes"])
        )
        
    def forward(self, x):
        return self.base_model(x)

def get_model(config=MODEL_CONFIG, device=None):
    """Get model instance and move to specified device."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DiabeticRetinopathyModel(config)
    model = model.to(device)
    return model 