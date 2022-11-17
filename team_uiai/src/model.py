import torch
import torch.nn as nn
import torch.nn.functional as F

# https://arxiv.org/abs/1610.00087
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 3 x 500 x 500 input RGB 3 channels
        # 2 x 500 x 500 output with 2 classes

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(8, 12, 2, stride=1, padding=1), 
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(12, 16, 2, stride=1, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(32, 64, 7), 
            nn.BatchNorm2d(64),
        )


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 12, 2, stride=2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 8, 4, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 2, 2, stride=1)
        ) 

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return {"out": x}