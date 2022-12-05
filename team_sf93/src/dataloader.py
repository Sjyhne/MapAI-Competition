import pathlib

from torch.utils.data import Dataset, DataLoader
from yaml import load, Loader
from datasets import load_dataset
import os
import torch
import cv2 as cv
import numpy as np


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


def load_image(imagepath: str, size: tuple) -> np.ndarray:
    image = cv.imread(imagepath, cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, size)
    
    # image = torch.tensor(image.astype(np.uint8)) / 255
    # image = torch.permute(image, (2, 0, 1))
    image = image.astype(np.uint8) / 255
    image = np.array(image, dtype="float32")
    
    return image


def load_label(labelpath: str, size: tuple) -> np.ndarray:
    label = cv.imread(labelpath, cv.IMREAD_GRAYSCALE)
    label[label == 255] = 1
    label = cv.resize(label, size)
    
    # label = torch.tensor(label.astype(np.uint8)).long()
    label = label.astype(np.uint8)
    
    return label


def load_lidar(lidarpath: str, size: tuple) -> torch.tensor:
    lidar = cv.imread(lidarpath, cv.IMREAD_UNCHANGED)
    lidar = cv.resize(lidar, size)

    # lidar = torch.tensor(lidar.astype(np.float)).float()
    lidar = lidar.astype(float)

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
                 datatype: str = "validation",
                 transform = None):

        self.opts = opts
        self.paths = download_dataset(data_type=datatype, task=opts["task"], get_dataset=True)
        self.transform = transform

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
        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']

        assert image.shape[1:] == label.shape[
                                  :2], f"image and label shape not the same; {image.shape[1:]} != {label.shape[:2]}"

        return image, label, filename

from albumentations.pytorch import ToTensorV2

class ImageLabelAndLidarDataset(Dataset):

    def __init__(self,
                 opts: dict,
                 datatype: str = "validation",
                 transform = None,
                 image_only_transforms = None):

        self.opts = opts

        self.paths = download_dataset(data_type=datatype, task=opts["task"], get_dataset=True)
        self.transform = transform
        self.image_only_transforms = image_only_transforms
        self.to_tensor = ToTensorV2()

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

        # assert image.shape[1:] == label.shape[
        #                           :2], f"image and label shape not the same; {image.shape[1:]} != {label.shape[:2]}"
        # assert image.shape[1:] == lidar.shape[
        #                           :2], f"image and label shape not the same; {image.shape[1:]} != {label.shape[:2]}"

        # perform transforms on RGB image (Note: only use image adjustment transforms like HSV, brightness)
        if self.image_only_transforms is not None:
            transformed = self.image_only_transforms(image=image, mask=label)
            image = transformed['image']

        lidar = np.expand_dims(lidar, 2)
        image = np.concatenate((image, lidar), axis=2)
        
        # perform transforms on RGB + Lidar
        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']

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


def create_dataloader(opts: dict,  transform=None, image_only_transform=None, datatype: str = "test",) -> DataLoader:
    if opts["task"] == 1:
        dataset = ImageAndLabelDataset(opts, datatype, transform)
    elif opts["task"] == 2:
        dataset = ImageLabelAndLidarDataset(opts, datatype, transform, image_only_transform)

    dataloader = DataLoader(dataset, batch_size=opts[f"task{opts['task']}"]["batchsize"], shuffle=opts[f"task{opts['task']}"]["shuffle"])

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
