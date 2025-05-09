import torch
import torch.nn as nn
from torchvision import transforms

class EMNIST_CNN(nn.Module):
    # Model architecture (same as training)
    def __init__(self, num_classes=47):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),  # After 3 max pools of 2, 28x28 becomes 3x3
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  
        x = self.fc_layers(x)
        return x

def load_model(path='cnn_model.pth'):
    checkpoint = torch.load(path, map_location='cpu')
    model = EMNIST_CNN(num_classes=len(checkpoint['class_mapping']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['class_mapping']

# Add these imports at the top
import torchvision.transforms.functional as F

# Modify the transform to include rotation and flip
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])