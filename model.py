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
        #self.tkan = TKAN(input_dim=4096, hidden_dim=512, output_dim=124)
        self.tkan = TKAN(
              512,
              sub_kan_configs=['relu', 'relu', 'relu'],
              return_sequences=False,
              use_bias=True
          )


        self.final_cnn = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.BatchNorm2d(128),  # Added Batch Normalization
            nn.Dropout(0.3)       # Added Dropout
        )
        
        self.fc = nn.Linear(128, 124)  # 124 subject classes

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn_extractor(x)         # (B*T, 64, 16, 16)
        x = self.final_cnn(x)             # (B*T, 128, 1, 1)
        x = x.view(B, T, 128)             # (B, T, 128)
        x = self.tkan(x)                  # (B, 512)
        x = self.fc(x)                    # (B, 124)
        return x


