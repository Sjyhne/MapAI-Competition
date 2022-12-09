import pathlib

from torch.utils.data import Dataset, DataLoader
from yaml import load, Loader
from datasets import load_dataset
import os
import torch
import cv2 as cv
import numpy as np
import kornia.augmentation as K
import torch.nn as nn
from semseg.augmentations import *

# class MyAugmentationPipeline(nn.Module):
#    def __init__(self) -> None:
#       super(MyAugmentationPipeline, self).__init__()
#       self.aff = K.RandomAffine(360)
#       self.jit = K.ColorJiggle(0.2, 0.3, 0.2, 0.3)
#
#    def forward(self, input, mask):
#       assert input.shape == mask.shape,
#          f"Input shape should be consistent with mask shape, "
#          f"while got {input.shape}, {mask.shape}"
#
#       aff_params = self.aff.forward_parameters(input.shape)
#       input = self.aff(input, aff_params)
#       mask = self.aff(mask, aff_params)
#
#       jit_params = self.jit.forward_parameters(input.shape)
#       input = self.jit(input, jit_params)
#       mask = self.jit(mask, jit_params)
#       return input, mask

def aug_image_mask(image,label):
    # import ipdb; ipdb.set_trace()
    aug_both = Compose([
        RandomResizedCrop((512, 512)),
        RandomCrop((512, 512), p=1.0),
        Pad((512, 512))

    ])
    # aug_img= Compose([   ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1))
    #
    # ])
    label = label.unsqueeze(0)

    image, label = aug_both(image, label)
    label = label.squeeze(0)

    # image = aug_img(image)
    return image, label

def get_paths_from_folder(folder: str) -> list:
    allowed_filetypes = ["jpg", "jpeg", "png", "tif", "tiff"]

    paths = []

    for file in os.listdir(folder):
        filetype = file.split(".")[1]

        if filetype not in allowed_filetypes:
            continue

        path = os.path.join(folder, file)

        paths.append(path)

    return paths


def load_image(imagepath: str, size: tuple) -> torch.tensor:
    image = cv.imread(imagepath, cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, size)

    image = torch.tensor(image.astype(np.uint8)) / 255
    image = torch.permute(image, (2, 0, 1))

    return image



def load_label(labelpath: str, size: tuple) -> torch.tensor:
    label = cv.imread(labelpath, cv.IMREAD_GRAYSCALE)
    label[label == 255] = 1
    label = cv.resize(label, size)
    label = torch.tensor(label.astype(np.uint8)).long()

    return label


def load_lidar(lidarpath: str, size: tuple) -> torch.tensor:
    lidar = cv.imread(lidarpath, cv.IMREAD_UNCHANGED)
    lidar = cv.resize(lidar, size)

    lidar = torch.tensor(lidar.astype(np.float)).float()

    return lidar


def download_dataset(data_type: str, task: int, get_dataset: bool = False):
    if data_type == "test":
        paths = load_dataset("sjyhne/mapai_evaluation_data", split=f"task{str(task)}", use_auth_token=True)
    else:
        paths = load_dataset("sjyhne/mapai_training_data", split=data_type)

    if get_dataset:
        return paths

    single_path = pathlib.Path(paths[0]["image"]).parent.parent.absolute()
    return single_path

class ImageAndLabelDataset(Dataset):

    def __init__(self,
                 opts: dict,
                 datatype: str = "validation"):

        self.opts = opts

        self.paths = download_dataset(data_type=datatype, task=opts["task"], get_dataset=True)

        print()

        print(
            f"Using number of images in {datatype}dataset: {int(self.paths.num_rows * self.opts['data_ratio'])}/{self.paths.num_rows}")

    def __len__(self):
        return int(self.paths.num_rows * self.opts["data_ratio"])

    def __getitem__(self, idx):
        pathdict = self.paths[idx]

        imagefilepath = pathdict["image"]
        labelfilepath = pathdict["mask"]

        assert imagefilepath.split("/")[-1] == labelfilepath.split("/")[
            -1], f"imagefilename and labelfilename does not match; {imagefilepath.split('/')[-1]} != {labelfilepath.split('/')[-1]}"

        filename = imagefilepath.split("/")[-1]

        image = load_image(imagefilepath, (self.opts["imagesize"], self.opts["imagesize"]))
        label = load_label(labelfilepath, (self.opts["imagesize"], self.opts["imagesize"]))

        ## add aug

        image, label = aug_image_mask(image, label)


        assert image.shape[1:] == label.shape[
                                  :2], f"image and label shape not the same; {image.shape[1:]} != {label.shape[:2]}"

        return image, label, filename


class ImageLabelAndLidarDataset(Dataset):

    def __init__(self,
                 opts: dict,
                 datatype: str = "validation"):

        self.opts = opts

        self.paths = download_dataset(data_type=datatype, task=opts["task"], get_dataset=True)

        print(
            f"Using number of images in {datatype}dataset: {int(self.paths.num_rows * self.opts['data_ratio'])}/{self.paths.num_rows}")

    def __len__(self):
        return int(self.paths.num_rows * self.opts["data_ratio"])

    def __getitem__(self, idx):

        pathdict = self.paths[idx]

        imagefilepath = pathdict["image"]
        labelfilepath = pathdict["mask"]
        lidarfilepath = pathdict["lidar"]

        assert imagefilepath.split("/")[-1] == labelfilepath.split("/")[
            -1], f"imagefilename and labelfilename does not match; {imagefilepath.split('/')[-1]} != {labelfilepath.split('/')[-1]}"
        assert imagefilepath.split("/")[-1] == lidarfilepath.split("/")[
            -1], f"imagefilename and labelfilename does not match; {imagefilepath.split('/')[-1]} != {labelfilepath.split('/')[-1]}"

        filename = imagefilepath.split("/")[-1]

        image = load_image(imagefilepath, (self.opts["imagesize"], self.opts["imagesize"]))
        label = load_label(labelfilepath, (self.opts["imagesize"], self.opts["imagesize"]))

        # add augmentation TODO

        ## TODO
        lidar = load_lidar(lidarfilepath, (self.opts["imagesize"], self.opts["imagesize"]))

        assert image.shape[1:] == label.shape[
                                  :2], f"image and label shape not the same; {image.shape[1:]} != {label.shape[:2]}"
        assert image.shape[1:] == lidar.shape[
                                  :2], f"image and label shape not the same; {image.shape[1:]} != {label.shape[:2]}"

        # Concatenate lidar and image data
        lidar = lidar.unsqueeze(0)

        image = torch.cat((image, lidar), dim=0)

        return image, label, filename


class TestDataset(Dataset):
    def __init__(self,
                 opts: dict,
                 datatype: str = "test"):
        self.opts = opts

        self.imagepaths = get_paths_from_folder(opts[datatype]["imagefolder"])

        print(f"Number of images in {datatype}dataset: {len(self)}")

    def __len__(self):
        return len(self.imagepaths)

    def __getitem__(self, idx):
        imagefilepath = self.imagepaths[idx]

        filename = imagefilepath.split("/")[-1]

        image = load_image(imagefilepath, (self.opts["imagesize"], self.opts["imagesize"]))

        return image, filename


def create_dataloader(opts: dict, datatype: str = "test") -> DataLoader:
    if opts["task"] == 1:
        dataset = ImageAndLabelDataset(opts, datatype)
    elif opts["task"] == 2:
        dataset = ImageLabelAndLidarDataset(opts, datatype)

    dataloader = DataLoader(dataset, batch_size=opts[f"task{opts['task']}"]["batchsize"], shuffle=opts[f"task{opts['task']}"]["shuffle"])

    return dataloader



if __name__ == "__main__":

    opts = load(open("../config/data.yaml"), Loader=Loader)

    testloader = create_dataloader(opts, "test")

    for batch in testloader:
        image, label, filename = batch

        print("image.shape:", image.shape)
        print("label.shape:", label.shape)
        print("filename:", filename)

        exit()