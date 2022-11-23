import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn


class SkipConnectionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect')
        self.bn = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(256)
    
    def forward(self, x_input):
        x = self.conv(x_input)
        x = nn.functional.gelu(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = nn.functional.gelu(x)
        x = self.bn2(x)
        return x + x_input



# https://arxiv.org/abs/1610.00087
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 3 x 500 x 500 input RGB 3 channels
        # 2 x 500 x 500 output with 2 classes

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 8, 3, stride=2, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=1),
        #     nn.Conv2d(8, 12, 2, stride=1, padding=1), 
        #     nn.BatchNorm2d(12),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=1),
        #     nn.Conv2d(12, 16, 2, stride=1, padding=1), 
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=1),
        #     nn.Conv2d(16, 32, 3, stride=2, padding=1), 
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=1),
        #     nn.Conv2d(32, 64, 7), 
        #     nn.BatchNorm2d(64),
        # )
        self.encoder = maskrcnn_resnet50_fpn(pretrained_backbone=True, trainable_backbone_layers=5).backbone

        self.feature_merger = nn.Conv2d(256*4, 256, 1)

        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, 7),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(16, 12, 2, stride=2),
        #     nn.BatchNorm2d(12),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(12, 8, 4, stride=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(8, 2, 2, stride=1)
        # )
        self.decoder = nn.Sequential(
            # nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
            # nn.GELU(),
            # nn.BatchNorm2d(256),
            # nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
            # nn.GELU(),
            # nn.BatchNorm2d(256),
            # nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
            # nn.GELU(),
            # nn.BatchNorm2d(256),
            # nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
            # nn.GELU(),
            # nn.BatchNorm2d(256),
            SkipConnectionModule(),
            SkipConnectionModule(),
            #nn.ConvTranspose2d(256, 256, 4, stride=4),
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            nn.Conv2d(256, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.cat(
            (
                x['0'],
                nn.functional.upsample(x['1'], (125, 125), mode='bilinear', align_corners=True),
                nn.functional.upsample(x['2'], (125, 125), mode='bilinear', align_corners=True),
                nn.functional.upsample(x['3'], (125, 125), mode='bilinear', align_corners=True),
            ),
            dim = 1
            )
        x = self.feature_merger(x)
        x = (self.decoder(x) + 1.0) / 2.0
        #print(torch.max(x))
        return {"out": x}