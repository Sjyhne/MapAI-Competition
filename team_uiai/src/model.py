import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
import torchvision

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
        self.encoder = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnext101_32x8d',
                        weights=torchvision.models.resnet.ResNeXt101_32X8D_Weights.IMAGENET1K_V2, trainable_layers=5)

        self.feature_merger = nn.Conv2d(256*4, 256, 1)


        self.decoder = nn.Sequential(
            SkipConnectionModule(),
            SkipConnectionModule(),
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
        return {"out": x}
