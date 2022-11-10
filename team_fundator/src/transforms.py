

import warnings
import albumentations as A
import numpy as np
import random
warnings.simplefilter("ignore")


class LidarAugComposer():
    """
    Provides normalization and augmentation of lidar data
    """
    def __init__(self, opts):
        lidar_opts = opts["lidar_augs"]

        lidar_augs = {
            "dropout": self.dropout,
            "random_scaling": self.random_scaling,
            "random_offset": self.random_offset,
            "random_noise": self.random_noise,
        }

        self.aug_list = []

        if len(lidar_opts.get("other_augs", [])):
            for t_name in lidar_opts["other_augs"]:
                self.aug_list.append(lidar_augs[t_name])

        if len(lidar_opts.get("one_of", [])) > 0:
            self.one_of_transforms = [lidar_augs[t_name] for t_name in lidar_opts["one_of"]]
            self.aug_list.append(self.one_of)    


        self.clip_min = lidar_opts.get("clip_min", 0.0)
        self.clip_max = lidar_opts.get("clip_max", 30.0)
        self.norm = lidar_opts.get("norm", "max")
        self.norm_basis = lidar_opts.get("norm_basis", "clip")
        
        self.opts = lidar_opts

    def normalize_lidar(self, lidar: np.ndarray):
        if self.norm == "max":
            return self.max_lidar_transform(lidar)
        return self.min_max_lidar_transform(lidar)

    def max_lidar_transform(self, lidar: np.ndarray):
        lidar = np.clip(lidar, self.clip_min, self.clip_max)
        lidar = lidar / (self.clip_max if self.norm_basis == "clip" else np.max(lidar))
        return lidar

    def min_max_lidar_transform(self, lidar: np.ndarray):
        lidar = np.clip(lidar, self.clip_min, self.clip_max)
        minimum = self.clip_min if self.norm_basis == "clip" else np.min(lidar)
        maximum = self.clip_max if self.norm_basis == "clip" else np.min(lidar)
        return (lidar - minimum) / (maximum - minimum)

    def dropout(self, lidar: np.ndarray):
        dropout_opts = self.opts["dropout"]
        if random.random() <= dropout_opts["p"]:
            size = lidar.shape[1] * lidar.shape[0]
            apply_indices = np.random.choice(size, replace=False, size=int(size * dropout_opts["pixel_frac"]))

            if dropout_opts["min_replacement"] != dropout_opts["max_replacement"]:
                replace_vals = np.random.uniform(dropout_opts["min_replacement"], dropout_opts["max_replacement"], size=int(size * dropout_opts["pixel_frac"]))
            else:
                replace_vals = dropout_opts["min_replacement"]
            lidar[np.unravel_index(apply_indices, lidar.shape)] = replace_vals

        return lidar

    def random_scaling(self, lidar: np.ndarray):
        rs_opts = self.opts["random_scaling"]
        if random.random() <= rs_opts["p"]:
            scale = random.uniform(rs_opts["min_scale"], rs_opts["max_scale"])
            return lidar * scale
        return lidar

    def random_noise(self, lidar: np.ndarray):
        rn_opts = self.opts["random_noise"]
        if random.random() <= rn_opts["p"]:
            noise = np.random.normal(scale=rn_opts["std"], size=lidar.shape)

            if rn_opts["keep_zero"]:
                noise[lidar == 0.0] = 0.0

            lidar += noise
        return lidar

    def random_offset(self, lidar: np.ndarray):
        ro_opts = self.opts["random_offset"]
        if random.random() <= ro_opts["p"]:
            offset = random.uniform(ro_opts["min_offset"], ro_opts["max_offset"])
            lidar[lidar >= ro_opts["min_height"]] += offset
        return lidar
        
    def one_of(self, lidar: np.ndarray):
        r = random.random()
        for i, transform in enumerate(self.one_of_transforms):
            if r <= (i + 1) / len(self.one_of_transforms):
                return transform(lidar)
    
    def lidar_transform(self, lidar: np.ndarray):
        for transform in self.aug_list:
            lidar = transform(lidar)
        return self.normalize_lidar(lidar)
    
    def get_transforms(self):
        return self.lidar_transform, self.normalize_lidar

# -------------------------------------------------------------------------------------------
#  orignially from https://github.com/qubvel/open-cities-challenge/blob/master/src/datasets/transforms.py
# -------------------------------------------------------------------------------------------

# --------------------------------------------------------------------
# Helpful functions
# --------------------------------------------------------------------

def p_transform(image, **kwargs):
    if image.ndim == 3:
        return image.transpose(2, 0, 1).astype("float32")
    else:
        return image.astype("float32")


# --------------------------------------------------------------------
# Segmentation transforms
# --------------------------------------------------------------------

post_transform = A.Lambda(name="post_transform", image=p_transform, mask=p_transform)


# crop 512
def normal(image_size):
    return A.Compose([
        #A.RandomCrop(512, 512, p=1.),
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

def task3_and_4_augs(image_size):
    return A.Flip(p=0.75)


# crop 768 (original) and hard augs
# ommited stage: crop 1024 (original) and same hard augs
def hard(image_size):
    return A.Compose([
        A.RandomScale(scale_limit=0.3, p=0.5),
        A.PadIfNeeded(image_size, image_size, p=1),
        A.RandomCrop(image_size, image_size, p=1.),
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
def very_hard(image_size):
    return A.Compose([
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=45, border_mode=0, value=0, p=0.7),
        A.PadIfNeeded(image_size, image_size, border_mode=0, value=0, p=1.),
        A.RandomCrop(image_size, image_size, p=1.),
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