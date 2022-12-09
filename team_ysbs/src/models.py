# -*- coding: utf-8 -*-
import segmentation_models_pytorch as smp
from pytorch_toolbelt.modules import decoders as D
from pytorch_toolbelt.modules import encoders as E
from torch import nn

from transunet import TransUNet


class SEResNeXt50FPN(nn.Module):
    def __init__(self, num_classes, fpn_channels=3):
        super().__init__()
        self.encoder = E.SEResNeXt50Encoder()
        self.decoder = D.FPNCatDecoder(self.encoder.channels, fpn_channels)
        self.logits = nn.Conv2d(self.decoder.channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.logits(x[0])


def load_model(model_name='unet', opts=None):
    out = 0
    aux_params = dict(
        pooling="max",
        dropout=0.3,
        activation="sigmoid",
        classes=opts["num_classes"],
    )
    if 'unet' == model_name:
        model = smp.Unet(encoder_name="resnet101", encoder_weights="imagenet", classes=opts["num_classes"], aux_params=aux_params)
    elif 'fpn' == model_name:
        model = smp.FPN(encoder_name="resnet101", encoder_weights="imagenet", classes=opts["num_classes"], aux_params=aux_params)
    elif "transunet" == model_name:
        out = -1
        in_c = 3
        if opts["task"] == 2:
            in_c = 4

        model = TransUNet(img_dim=opts['imagesize'],
                          in_channels=in_c,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=opts["num_classes"])

    return model, out


def main():
    pass


if __name__ == "__main__":
    main()
