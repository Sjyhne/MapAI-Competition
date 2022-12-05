''' 
References; 

https://arxiv.org/abs/2102.04306
https://www.sciencedirect.com/science/article/pii/S1361841518306133
https://arxiv.org/abs/1803.08494
https://arxiv.org/abs/1505.04597
https://arxiv.org/abs/2010.11929
''' 

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from einops import rearrange

from vit import ViT


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients):
        super(AttentionBlock, self).__init__()
        self.w_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(8, n_coefficients)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(8, n_coefficients)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )
        self.gelu = nn.GELU()

    def forward(self, gate, skip_connection):
        g1 = self.w_gate(gate)
        x1 = self.w_x(skip_connection)
        psi = self.gelu(g1 + x1)
        psi = self.psi(psi)
        return skip_connection * psi


class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.GroupNorm(8, out_channels)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.GroupNorm(8, width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.GroupNorm(8, width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.GroupNorm(8, out_channels)

        self.gelu = nn.GELU()

        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x):
        x_down = self.down_sample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.gelu(x)
        x = self.dropout(x)

        return x


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up_sample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.attention_gate = AttentionBlock(F_g=int(in_channels // 2), F_l=int(in_channels // 2), n_coefficients=int(out_channels))

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    def forward(self, x, skip=None):
        x = self.up_sample(x)
        if skip is not None:
            ag = self.attention_gate(gate=x, skip_connection=skip)
            x = torch.cat([ag, x], dim=1)

        return self.layer(x)


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.gelu = nn.GELU()

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        self.vit_img_dim = img_dim // patch_dim
        self.vit = ViT(self.vit_img_dim, out_channels * 8, out_channels * 8,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        out_final_channel = int((out_channels * 8) / 2)
        self.conv2 = nn.Conv2d(out_channels * 8, out_final_channel, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(8, out_final_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.gelu(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)

        x = self.vit(x)
        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.gelu(x)

        return x, x1, x2, x3


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)

        return x


class TransUNet(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, class_num):
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_dim)

        self.decoder = Decoder(out_channels, class_num)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x = self.decoder(x, x1, x2, x3)

        return x


if __name__ == '__main__':
    import torch

    transunet = TransUNet(img_dim=512,
                          in_channels=3,
                          out_channels=64,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=1)

    print(sum(p.numel() for p in transunet.parameters()))
    print(transunet(torch.randn(1, 3, 512, 512)).shape)
