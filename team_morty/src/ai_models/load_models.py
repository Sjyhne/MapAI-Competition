import torchvision
import torch.nn as nn
import segmentation_models_pytorch as smp
from ai_models.utils import set_parameter_requires_grad


def load_unet(opts):
    get_output = 0
    aux_params = dict(
        pooling="avg",  # one of 'avg', 'max'
        dropout=0.5,  # dropout ratio, default is None
        activation="sigmoid",  # activation function, default is None
        classes=opts["num_classes"],  # define number of output labels
    )
    return (
        smp.Unet("resnet50", classes=opts["num_classes"], aux_params=aux_params),
        get_output,
    )


def load_resnet50(
    opts,
    pretrained=False,
    freeze=False,
    pretrained_model="FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1",
):
    get_output = "out"
    if not pretrained:
        return (
            torchvision.models.segmentation.fcn_resnet50(
                pretrained=False, num_classes=opts["num_classes"]
            ),
            get_output,
        )
    model = torchvision.models.segmentation.fcn_resnet50(
        pretrained=pretrained_model, num_classes=opts["num_classes"]
    )
    # set_parameter_requires_grad(model, True)
    num_ftrs = model.aux_classifier[4].in_channels
    model.aux_classifier[4] = nn.Conv2d(
        num_ftrs, opts["num_classes"], kernel_size=(1, 1)
    )
    return model, get_output


def load_resnet101(opts, pretrained=False):
    get_output = "out"
    return (
        torchvision.models.segmentation.fcn_resnet101(
            pretrained=False, num_classes=opts["num_classes"]
        ),
        get_output,
    )


# The current model should be swapped with a different one of your choice
# model = torchvision.models.segmentation.fcn_resnet50(pretrained="FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1")
# aux_params=dict(
# pooling='avg',             # one of 'avg', 'max'
# dropout=0.5,               # dropout ratio, default is None
# activation='sigmoid',      # activation function, default is None
# classes=opts["num_classes"],                 # define number of output labels
# )
# model = smp.Unet('resnet50', classes=opts["num_classes"], aux_params=aux_params)#, decoder_channels = (512, 256, 128, 64, 32, 16), encoder_depth=6)

# decoder_channels: List[int] = (256, 128, 64, 32, 16),
# mask, label = model(x)

# model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=opts["num_classes"])
# model = torchvision.models.segmentation.fcn_resnet101(pretrained=False, num_classes=opts["num_classes"] )
# set_parameter_requires_grad(model, True)
# num_ftrs = model.aux_classifier[4].in_channels
# model.aux_classifier[4] = nn.Conv2d(num_ftrs, opts["num_classes"], kernel_size=(1, 1))
# model = torchvision.models.segmentation.fcn_resnet50(pretrained="FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1", num_classes=opts["num_classes"])
# model = UNet()
