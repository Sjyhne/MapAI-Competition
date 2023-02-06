"""
@File           : UNet.py
@Author         : Gefei Kong
@Time:          : 18.10.2022 15:27
------------------------------------------------------------------------------------------------------------------------
@Description    : UNet model for semantic segmentation

related links:
1. UNet --> https://arxiv.org/abs/1505.04597
2. Pytorch UNet codes --> https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    convolutional block for UNet:
        conv2D+BN+ReLU
        conv2D+BN+ReLU
    """
    def __init__(self, in_channel, out_channel, second_channel):
        super().__init__()

        # note: bias = False
        # reason: the conv block is "conv2D+BN+ReLU".
        #         Considering the usage of BN, bias is not necessary and will affect code efficiency
        # more detailed explanation: https://blog.csdn.net/u013289254/article/details/98785869
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channel, second_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(second_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(second_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class DownSamplingBlock(nn.Module):
    """
    One down sampling block:
        maxpooling + conv_block
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.down_block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channel, out_channel, second_channel=out_channel)
        )

    def forward(self, x):
        return self.down_block(x)


class UpSamplingBlock(nn.Module):
    """
    One up sampling block:
        convTranspose2D + concat + conv_block
    """

    def __init__(self, in_channel, out_channel):
        super().__init__()
        # please note:
        self.up_T2d  = nn.ConvTranspose2d(in_channel, out_channels=in_channel // 2, kernel_size=2, stride=2)
        self.up_conv = ConvBlock(in_channel, out_channel, second_channel=out_channel)

    def forward(self, x1, x2):
        x1 = self.up_T2d(x1)
        # ==============================================================================================================
        # match padding size
        # input type: B*C*H*W
        # reference:
        #   https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
        # other references:
        #   if you have padding issues, see
        #   https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        #   https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # may have userwarnings
        # because // may cause problem when facing negative values
        # more details:
        # https://pytorch.org/docs/stable/generated/torch.div.html
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])


        # ==============================================================================================================

        # note: there is x1, x2， rather than x2, x1
        # which is different with original UNet
        # x1: downsampling_conv result at this layer, with more features before convTranspose2D
        # x2: the skip_connection features, the downsampling_conv result at previous layer, with less features
        # original UNet:  skip_decoder_res + encoder_res
        # here:           encoder_res  + skip_decoder_res
        x = torch.cat([x1, x2], dim=1)
        x = self.up_conv(x)

        return x


class UNet(nn.Module):
    """
    UNet
    """
    def __init__(self, inp_channel, n_cls):
        super().__init__()
        self.inp_channel = inp_channel
        self.n_cls = n_cls

        # downsampling
        self.inpl  = ConvBlock(inp_channel, 64, second_channel=64) #  conv_block:   "Conv+BN+ReLU" * 2

        self.down1 = DownSamplingBlock(64, 128)     # maxpooling + conv_block
        self.down2 = DownSamplingBlock(128, 256)    # maxpooling + conv_block
        self.down3 = DownSamplingBlock(256, 512)    # maxpooling + conv_block
        self.down4 = DownSamplingBlock(512, 1024)   # maxpooling + conv_block

        # upsampling
        self.up1   = UpSamplingBlock(1024, 512)     # ConvTranspose2D + concat + conv_block
        self.up2   = UpSamplingBlock(512, 256)      # ConvTranspose2D + concat + conv_block
        self.up3   = UpSamplingBlock(256, 128)      # ConvTranspose2D + concat + conv_block
        self.up4   = UpSamplingBlock(128, 64)       # ConvTranspose2D + concat + conv_block

        # outconv
        self.outconv = nn.Conv2d(64, n_cls, kernel_size=1)  # conv

    def forward(self, x):
        # downsampling
        x_in = self.inpl(x)         # in_channels -> (B, 64, H, W)
        x_d1 = self.down1(x_in)     # 64    -> (B, 128, H, W)
        x_d2 = self.down2(x_d1)     # 128   -> (B, 256, H, W)
        x_d3 = self.down3(x_d2)     # 256   -> (B, 512, H, W)
        x_d4 = self.down4(x_d3)     # 512   -> (B, 1024, H, W)

        # upsampling
        x_u1 = self.up1(x_d4, x_d3)  # 1024  -> 512
        x_u2 = self.up2(x_u1, x_d2)  # 512   -> 256
        x_u3 = self.up3(x_u2, x_d1)  # 256   -> 128
        x_u4 = self.up4(x_u3, x_in)  # 128   -> 64

        out  = self.outconv(x_u4)    # 64    -> n_classes

        return out


if __name__=='__main__':
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = UNet(inp_channel=3, n_cls=2).to(device)

    # print network structure
    from torchinfo import summary
    summary(model, (1,3,572,572))

    # # visualize
    # import netron
    #
    # tmp_in = torch.randn(1, 3, 572, 572, device=device)  # Randomly generate an input
    # modelData = '../../../output/model_test/demo.onnx'
    # torch.onnx.export(model, tmp_in, modelData)
    #
    # netron.start(modelData)











