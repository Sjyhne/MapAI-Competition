
# -------------------------------------------------------------------------------------------
#  orignially from https://github.com/qubvel/open-cities-challenge/blob/master/src/datasets/transforms.py
# -------------------------------------------------------------------------------------------
import warnings
import albumentations as A
import numpy as np

warnings.simplefilter("ignore")

# --------------------------------------------------------------------
# Helpful functions
# --------------------------------------------------------------------

def post_transform(image, **kwargs):
    if image.ndim == 3:
        return image.transpose(2, 0, 1).astype("float32")
    else:
        return image.astype("float32")


# --------------------------------------------------------------------
# Segmentation transforms
# --------------------------------------------------------------------

post_transform = A.Lambda(name="post_transform", image=post_transform, mask=post_transform)

def get_lidar_transform(opts):
    lidar_augs = opts["lidar_augs"]
    clip_min = lidar_augs.get("clip_min", 0.0)
    clip_max = lidar_augs.get("clip_max", 30.0)
    norm = lidar_augs.get("norm", "max")
    norm_basis = lidar_augs.get("norm_basis", "clip")

    if norm not in ["min_max", "max"]:
        print(f"Norm {norm} not recognized. Can only normalize with clip_max or image_max")
        exit()
    if norm_basis not in ["clip", "image"]:
        print(f"Norm_basis {norm} not recognized. Can only normalize with clip values or image maxima/minima")
        exit()
    
    if norm == "max":
        def max_lidar_transform(lidar):
            lidar = np.clip(lidar, clip_min, clip_max)
            lidar = lidar / (clip_max if norm_basis == "clip" else np.max(lidar))
            return lidar
        return max_lidar_transform
        
    def min_max_lidar_transform(lidar):
        lidar = np.clip(lidar, clip_min, clip_max)
        minimum = clip_min if norm_basis == "clip" else np.min(lidar)
        maximum = clip_max if norm_basis == "clip" else np.min(lidar)
        lidar = (lidar - minimum) / (maximum - minimum)
        return lidar
    return min_max_lidar_transform


# crop 512
normal = A.Compose([
    A.RandomCrop(512, 512, p=1.),
    A.Flip(p=0.75),
    A.RandomBrightnessContrast(p=0.5),
    post_transform,
])

valid_transform = A.Compose([
    post_transform,
])

test_transform = A.Compose([
    post_transform,
])

# crop 768 (original) and hard augs
# ommited stage: crop 1024 (original) and same hard augs
hard = A.Compose([
    A.RandomScale(scale_limit=0.3, p=0.5),
    A.PadIfNeeded(512, 512, p=1),
    A.RandomCrop(512, 512, p=1.),
    A.Flip(p=0.75),
    A.Downscale(scale_min=0.5, scale_max=0.75, p=0.05),

    # color transforms
    A.OneOf(
        [
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
            A.ChannelShuffle(p=0.2),
            A.HueSaturationValue(p=1),
            A.RGBShift(p=1),
        ],
        p=0.5,
    ),

    # noise transforms
    A.OneOf(
        [
            A.GaussNoise(p=1),
            A.MultiplicativeNoise(p=1),
            #A.IAASharpen(p=1),
            # A.ImageCompression(quality_lower=0.7, p=1),
            A.GaussianBlur(p=1),
        ],
        p=0.2,
    ),
    post_transform,
])


# crop 768 (original) and very hard augs
very_hard = A.Compose([
    A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=45, border_mode=0, value=0, p=0.7),
    A.PadIfNeeded(512, 512, border_mode=0, value=0, p=1.),
    A.RandomCrop(512, 512, p=1.),
    A.Flip(p=0.75),
    A.Downscale(scale_min=0.5, scale_max=0.75, p=0.05),
    A.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0, p=0.1),

    # color transforms
    A.OneOf(
        [
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.RandomGamma(gamma_limit=(70, 130), p=1),
            A.ChannelShuffle(p=0.2),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1),
        ],
        p=0.8,
    ),

    # distortion
    A.OneOf(
        [
            A.ElasticTransform(p=1),
            A.OpticalDistortion(p=1),
            A.GridDistortion(p=1),
            #A.IAAPerspective(p=1),
        ],
        p=0.2,
    ),

    # noise transforms
    A.OneOf(
        [
            A.GaussNoise(p=1),
            A.MultiplicativeNoise(p=1),
            #A.IAASharpen(p=1),
            A.GaussianBlur(p=1),
        ],
        p=0.2,
    ),
    post_transform,
])