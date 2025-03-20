import torch
import torch.nn as nn
from tkan import TKAN  # Import TKAN

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x):
        return self.conv_layers(x)

class GaitRecognitionModel(nn.Module):
    def __init__(self):
        super(GaitRecognitionModel, self).__init__()
        self.cnn_extractor = CNNFeatureExtractor()
        self.tkan = TKAN(input_dim=4096, hidden_dim=512, output_dim=124)
        
        self.final_cnn = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.BatchNorm2d(128),  # Added Batch Normalization
            nn.Dropout(0.3)       # Added Dropout
        )
        
        self.fc = nn.Linear(128, 124)  # 124 subject classes

    def forward(self, x):
        x = self.cnn_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.tkan(x)
        x = x.view(x.size(0), 64, 1, 1)
        x = self.final_cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # Removed softmax, as CrossEntropyLoss applies it automatically
