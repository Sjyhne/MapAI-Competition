"""
@File           : E_DNet.py
@Author         : Gefei Kong
@Time:          : 23.10.2022 20:13
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
E-PA Net for building segmentation (v3)
Reference paper: https://ieeexplore.ieee.org/abstract/document/9408358

The structure defined by ourselves
compared with E_DNet, instead of the "E-Net -> D-Net -> Fusion" structure
In the E-PA Net, the new structure is:

Encoder -> parallel 3 dalitaed convs -> Decoder ---> convblock1 --> 3 classes (0-back 1-edge)
    |----> 1024 convs                --->|      |--> + convblock1 -> convblock2 --> 2 classes (0-back 1-fore)

Position of main changes: ~L122
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .UNet import ConvBlock, DownSamplingBlock, UpSamplingBlock, UNet

class DilateConvBlock_EPANet(nn.Module):
    """
    Three layers' dilated convolutions in E-Net
    """
    def __init__(self, in_channel, out_channel, dilated_rate=[1,2,5]):
        super().__init__()
        self.dconv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3,
                                padding=dilated_rate[0], dilation=dilated_rate[0], bias=False)

        self.dconv2 = nn.Conv2d(in_channel, out_channel, kernel_size=3,
                                padding=dilated_rate[1], dilation=dilated_rate[1], bias=False)

        self.dconv3 = nn.Conv2d(in_channel, out_channel, kernel_size=3,
                                padding=dilated_rate[2], dilation=dilated_rate[2], bias=False)

        # self.conv1_1 = nn.Conv2d(out_channel*3, out_channel, kernel_size=1)

    def forward(self, x):
        x1 = self.dconv1(x)
        x2 = self.dconv2(x)
        x3 = self.dconv3(x)

        # out = self.conv1_1(torch.cat([x1,x2,x3], dim=1))

        return torch.cat([x1,x2,x3], dim=1)


class DownBlockEDNet(nn.Module):
    """
    One down sampling block for E-D Net:
        conv_block + maxpooling
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.convb = ConvBlock(in_channel, out_channel, second_channel=out_channel)
        self.maxp  = nn.MaxPool2d(2)

    def forward(self, x):
        xc = self.convb(x)
        xp = self.maxp(xc)
        return xc, xp

class catConv_EPANet(nn.Module):
    """
    One up sampling block:
        convTranspose2D + concat + conv_block
    """

    def __init__(self, in_channel_bdge, in_channel_dilated, out_channel):
        super().__init__()
        self.up_T2d = nn.ConvTranspose2d(in_channel_bdge, out_channels=in_channel_bdge//2, kernel_size=2, stride=2)
        self.up_conv = ConvBlock(in_channel_bdge//2+in_channel_dilated, out_channel, second_channel=out_channel)

    def forward(self, x1, x2):
        """
        :param x1: [B, 1024,32,32] - bridge conv
        :param x2: [B, 512,64,64] - dilated conv blocks
        :return:
        """
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


class UpConvl1_EPANet(nn.Module):
    """
    One up sampling block:
        convTranspose2D + concat + conv_block
    """

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.up_conv = ConvBlock(out_channel*2, out_channel, second_channel=out_channel)

    def forward(self, x1, x2):
        x1 = self.up_conv1(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.up_conv(x)

        return x


class E_PANet_v3(nn.Module):
    def __init__(self, inp_channel, n_cls):
        super(E_PANet_v3, self).__init__()

        self.inp_channel = inp_channel
        self.n_cls = n_cls

        # downsampling
        self.down1 = DownBlockEDNet(inp_channel, 64)  # conv_block + maxpooling
        self.down2 = DownBlockEDNet(64, 128)   # conv_block + maxpooling
        self.down3 = DownBlockEDNet(128, 256)  # conv_block + maxpooling
        self.down4 = DownBlockEDNet(256, 512)  # conv_block + maxpooling

        # dilated convs -> 3 parallel dilated convs
        self.dilatedc = DilateConvBlock_EPANet(512, 512, dilated_rate=[1, 2, 5])

        # 1024 convs
        self.bdge_conv1024 = ConvBlock(512, 1024, second_channel=1024)

        # cat
        self.cat_up = catConv_EPANet(1024, 512*3, 1024)  # ConvTranspose2D + concat + conv_block

        # upsampling
        self.up1 = UpConvl1_EPANet(1024, 512)  # ConvTranspose2D + concat + conv_block
        self.up2 = UpSamplingBlock(512, 256)  # ConvTranspose2D + concat + conv_block
        self.up3 = UpSamplingBlock(256, 128)  # ConvTranspose2D + concat + conv_block
        self.up4 = UpSamplingBlock(128, 64)  # ConvTranspose2D + concat + conv_block

        # ==============================================================================================================
        # compared with E_PANet.py, changed part
        # outconv for 2 classes （b+e）
        self.midconv_b_e = ConvBlock(64, 64, second_channel=64)
        self.outconv_b_e = nn.Conv2d(64, 2, kernel_size=1)  # out_channels = 2-> foreground, edge layers

        # outconv for 2 classes
        # middle layer
        self.midconv_b_f = ConvBlock(128, 64, second_channel=64) # 128 = self.up4 + self.midconv_3cls
        self.outconv_b_f = nn.Conv2d(64, self.n_cls, kernel_size=1)  # out_channels = 2 -> foreground, background
        # ==============================================================================================================

    def forward(self, x):
        # downsampling
        x_c1, x_d1 = self.down1(x)      # 3     -> (B, 64, H, W)    x_c1: (H,W),     x_d1: (H/2,W/2)    512,256
        x_c2, x_d2 = self.down2(x_d1)   # 64    -> (B, 128, H, W)   x_c2: (H/2,W/2), x_d2: (H/4,W/4)    256,128
        x_c3, x_d3 = self.down3(x_d2)   # 128   -> (B, 256, H, W)   x_c3: (H/4,W/4), x_d3: (H/8,W/8)    128,64
        x_c4, x_d4 = self.down4(x_d3)   # 256   -> (B, 512, H, W)   x_c4: (H/8,W/8), x_d4: (H/16,W/16)  64,32

        # dilated convs
        x_dc = self.dilatedc(x_c4)      # 512   -> (B, 512*3, H, W)   x_dc: (H/8,W/8) 64

        # bridge conv
        x_bdgec = self.bdge_conv1024(x_d4) # 1024  -> (B, 512, H, W)   x_bc: (H/16, W/16) (similar to x_d4) 32

        # cat
        # x_bc([B,1024, H/16, W/16]) -(T2d)-> [B, 512, H/8, W/8] -cat(x_dc)--> [B, 512*4, H/8, W/8] -conv-> [B,1024,H,W]
        x_d_b = self.cat_up(x_bdgec, x_dc) # 1024  -> (B, 1024,H, W)   x_d_b: (H/8,W/8) 64

        # upsampling
        x_u1 = self.up1(x_d_b, x_c4)  # 1024  -> 512 x_d_b: (B, 1024,H/8,W/8), x_c4: (B, 512, H/8,W/8) 64
        x_u2 = self.up2(x_u1, x_c3)   # 512   -> 256 x_u1:  (B, 512, H/8,W/8), x_c3: (B, 256, H/4,W/4) 128
        x_u3 = self.up3(x_u2, x_c2)   # 256   -> 128 x_u2:  (B, 256, H/4,W/4), x_c2: (B, 128, H/2,W/2) 256
        x_u4 = self.up4(x_u3, x_c1)   # 128   -> 64  x_u3:  (B, 128, H/2,W/2), x_c1: (B, 64,  H,  W  ) 512 x_u4: (B, 64, H, W)

        # ==============================================================================================================
        # compared with E_PANet.py, changed part
        # out conv 3 classes
        mid_b_e = self.midconv_b_e(x_u4)  # 64 -> 64
        out_b_e = self.outconv_b_e(mid_b_e)  # 64    -> 2

        # out conv 2 classes
        mid_2cls  = self.midconv_b_f(torch.cat([x_u4, mid_b_e], dim=1)) # 64+64   -> 64
        out_2cls = self.outconv_b_f(mid_2cls) # 64  -> 2
        # ==============================================================================================================

        # fuse
        out_b_e_sof = F.softmax(out_b_e, dim=1).float()  # [B, 3, H, W]
        out_b_e_sof_edge = out_b_e_sof[:, 1, :, :]
        x_fuse_out_part1 = out_b_e_sof_edge[:, None, :, :] * out_2cls
        x_fuse_out = out_2cls + x_fuse_out_part1

        return out_b_e, out_2cls, x_fuse_out



if __name__=='__main__':
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # model = ENet(inp_channel=3).to(device)
    model = E_PANet_v3(inp_channel=3, n_cls=2).to(device)

    # print network structure
    from torchinfo import summary
    summary(model, (1,3,572,572))
