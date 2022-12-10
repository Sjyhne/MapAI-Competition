import os
import pathlib

import cv2 as cv
import numpy as np
import segmentation_models_pytorch as smp
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from yaml import load, Loader

from albumentation import *

train_augmentation = get_training_augmentation()
validation_augmentation = get_validation_augmentation()
preprocessing_fn = smp.encoders.get_preprocessing_fn("resnet101", "imagenet")
train_preprocess = get_preprocessing(preprocessing_fn)


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

    return image


def load_label(labelpath: str, size: tuple) -> torch.tensor:
    label = cv.imread(labelpath, cv.IMREAD_GRAYSCALE)
    label[label == 255] = 1
    label = cv.resize(label, size)

    # label = torch.tensor(label.astype(np.uint8)).long()

    return label


def load_lidar(lidarpath: str, size: tuple) -> torch.tensor:
    lidar = cv.imread(lidarpath, cv.IMREAD_UNCHANGED)
    lidar = cv.resize(lidar, size)
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
        self.datatype = datatype

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

        if self.datatype == "train" and self.opts['train']['augmentations']:
            # apply augmentations
            sample_aerial = train_augmentation(image=image, mask=label)
            image, label = sample_aerial['image'], sample_aerial['mask']
        # apply preprocessing
        if self.datatype == "train" and self.opts['train']['preprocessing']:
            sample_aerial = train_preprocess(image=image, mask=label)
            image, label = sample_aerial['image'], sample_aerial['mask']

        image = torch.tensor(image.astype(np.uint8)) / 255
        image = torch.permute(image, (2, 0, 1))
        label = torch.tensor(label.astype(np.uint8), device=self.opts["device"]).long()

        assert image.shape[1:] == label.shape[
                                  :2], f"image and label shape not the same; {image.shape[1:]} != {label.shape[:2]}"

        return image, label, filename


class ImageLabelAndLidarDataset(Dataset):

    def __init__(self,
                 opts: dict,
                 datatype: str = "validation"):
        self.opts = opts
        self.datatype = datatype

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
        lidar = load_lidar(lidarfilepath, (self.opts["imagesize"], self.opts["imagesize"]))

        # apply augmentations
        if self.datatype == "train" and self.opts['train']['augmentations']:
            sample = train_augmentation(image=image, lidar=lidar, mask=label)
            image, lidar, label = sample['image'], sample['lidar'], sample['mask']
        # apply preprocessing
        if self.datatype == "train" and self.opts['train']['preprocessing']:
            sample = self.preprocessing(image=image, lidar=lidar, mask=label)
            image, lidar, label = sample['image'], sample['lidar'], sample['mask']

        image = torch.tensor(image.astype(np.uint8)) / 255
        image = torch.permute(image, (2, 0, 1))

        lidar = torch.tensor(lidar.astype(np.float)).float()
        label = torch.tensor(label.astype(np.uint8), device=self.opts["device"]).long()

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


import matplotlib.pyplot as plt


# helper function for data visualization
def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap="gray")
    plt.show()


def create_dataloader(opts: dict, datatype: str = "test") -> DataLoader:
    if opts["task"] == 1:
        dataset = ImageAndLabelDataset(opts, datatype)
    elif opts["task"] == 2:
        dataset = ImageLabelAndLidarDataset(opts, datatype)

    dataloader = DataLoader(dataset, batch_size=opts[f"task{opts['task']}"]["batchsize"], shuffle=opts[f"task{opts['task']}"]["shuffle"])

    # visualize some augmented images
    for id in range(8):
        data = dataset[id]  # get some sample
        visualize(
            image=data[0].permute(1, 2, 0),
            mask=data[1].cpu().squeeze()
        )

    return dataloader


if __name__ == "__main__":

    opts = load(open("config/massachusetts.yaml"), Loader=Loader)

    testloader = create_dataloader(opts, "test")

    for batch in testloader:
        image, label, filename = batch

        print("image.shape:", image.shape)
        print("label.shape:", label.shape)
        print("filename:", filename)

        exit()
