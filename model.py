import torch
import torch.nn as nn
from tkan_pytorch import TKAN

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
        self.tkan = TKAN(
            input_dim=128,  # Matches final_cnn output
            hidden_dim=512,
            sub_kan_configs=[3, 2, 1],  # Cubic, quadratic, linear splines
            return_sequences=False,
            use_bias=True
        )
        self.final_cnn = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3)
        )
        self.fc = nn.Linear(512, 124)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # [B*T, 1, 64, 64]
        x = self.cnn_extractor(x)  # [B*T, 64, 16, 16]
        x = self.final_cnn(x)  # [B*T, 128, 1, 1]
        x = x.view(B, T, 128)  # [B, T, 128]
        x = self.tkan(x)  # [B, 512]
        x = self.fc(x)  # [B, 124]
        return x

"""
    def forward(self, x):
        B, T, C, H, W = x.shape
        cnn_out = []
        for t in range(T):
            frame = x[:, t, :, :, :]                                # (B, 1, 64, 64)
            spatial_features = self.cnn_extractor(frame)           # (B, 64, 16, 16)
            spatial_features = self.final_cnn(spatial_features)    # (B, 128, 1, 1)
            spatial_features = spatial_features.view(B, 128)       # (B, 128)
            cnn_out.append(spatial_features)
    
        cnn_out = torch.stack(cnn_out, dim=1)                      # (B, T, 128)
        x = self.tkan(cnn_out)                                     # (B, 512)
        x = self.fc(x)                                             # (B, 124)
        return x
"""
