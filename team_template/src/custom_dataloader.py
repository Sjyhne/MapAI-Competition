import pathlib

from torch.utils.data import Dataset, DataLoader
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


def load_image(imagepath: str, size: tuple) -> torch.tensor:
    image = cv.imread(imagepath, cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, size)

    image = torch.tensor(image.astype(np.uint8)) / 255
    #image = torch.permute(image, (2, 0, 1)) done by albummentations

    return image.numpy()


def load_label(labelpath: str, size: tuple) -> torch.tensor:
    label = cv.imread(labelpath, cv.IMREAD_GRAYSCALE)
    label[label == 255] = 1
    label = cv.resize(label, size)

    label = label.astype(np.int32)

    return label


def load_lidar(lidarpath: str, size: tuple) -> torch.tensor:
    lidar = cv.imread(lidarpath, cv.IMREAD_UNCHANGED)
    lidar = cv.resize(lidar, size)

    lidar = torch.tensor(lidar.astype(np.float)).float()

    return lidar




class ImageAndLabelDataset(Dataset):

    def __init__(self,
                 opts: dict,
                 datatype: str = "validation",
                 transforms=None):

        self.opts = opts

        root = opts["data_dirs"]["root"]
        self.image_paths = sorted(pathlib.Path(f"{root}/{datatype}/{opts['data_dirs']['images']}").glob("*.tif"))
        self.mask_paths = sorted(pathlib.Path(f"{root}/{datatype}/{opts['data_dirs']['masks']}").glob("*.tif"))
        
        assert len(self.image_paths)  == len(self.mask_paths) 
        print()

        print(
            f"Using number of images in {datatype}dataset: {int(len(self.image_paths) * self.opts['data_ratio'])}/{len(self.image_paths) }")
        self.transform = transforms
    
    def __len__(self):
        return int(len(self.image_paths) * self.opts["data_ratio"])

    def __getitem__(self, idx):

        imagefilepath = self.image_paths[idx].as_posix()
        labelfilepath = self.mask_paths[idx].as_posix()


        assert imagefilepath.split("/")[-1] == labelfilepath.split("/")[
            -1], f"imagefilename and labelfilename does not match; {imagefilepath.split('/')[-1]} != {labelfilepath.split('/')[-1]}"

        filename = imagefilepath.split("/")[-1]

        image = load_image(imagefilepath, (self.opts["imagesize"], self.opts["imagesize"]))
        label = load_label(labelfilepath, (self.opts["imagesize"], self.opts["imagesize"]))
        
        assert image.shape[:2] == label.shape[
                                  :2], f"image and label shape not the same; {image.shape[:2]} != {label.shape[:2]}"

        sample = dict(
            image=image,
            mask=label,
        )
        
        if self.transform is not None:
            sample = self.transform(**sample)
        else:
            sample["image"] = sample["image"].transpose(2, 0, 1)
        return sample


class ImageLabelAndLidarDataset(Dataset):

    def __init__(self,
                 opts: dict,
                 datatype: str = "validation"):

        self.opts = opts

        root = opts["data_dirs"]["root"]
        self.lidar_paths = sorted(pathlib.Path(f"{root}/{datatype}/{opts['data_dirs']['lidar']}").glob("*.tif"))
        self.image_paths = sorted(pathlib.Path(f"{root}/{datatype}/{opts['data_dirs']['images']}").glob("*.tif"))
        self.mask_paths = sorted(pathlib.Path(f"{root}/{datatype}/{opts['data_dirs']['masks']}").glob("*.tif"))

        assert len(self.image_paths)  == len(self.mask_paths) 
        assert len(self.image_paths)  == len(self.lidar_paths) 
        print(
            f"Using number of images in {datatype}dataset: {int(len(self.image_paths) * self.opts['data_ratio'])}/{len(self.image_paths) }")

    def __len__(self):
        return int(len(self.image_paths) * self.opts["data_ratio"])

    def __getitem__(self, idx):
        imagefilepath = self.image_paths[idx].as_posix()
        labelfilepath = self.mask_paths[idx].as_posix()
        lidarfilepath = self.lidar_paths[idx].as_posix()

        assert imagefilepath.split("/")[-1] == labelfilepath.split("/")[
            -1], f"imagefilename and labelfilename does not match; {imagefilepath.split('/')[-1]} != {labelfilepath.split('/')[-1]}"
        assert imagefilepath.split("/")[-1] == lidarfilepath.split("/")[
            -1], f"imagefilename and labelfilename does not match; {imagefilepath.split('/')[-1]} != {labelfilepath.split('/')[-1]}"

        filename = imagefilepath.split("/")[-1]

        image = load_image(imagefilepath, (self.opts["imagesize"], self.opts["imagesize"]))
        label = load_label(labelfilepath, (self.opts["imagesize"], self.opts["imagesize"]))
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


def create_dataloader(opts: dict, datatype: str = "test", transforms=None) -> DataLoader:
    if opts["task"] == 1:
        dataset = ImageAndLabelDataset(opts, datatype, transforms)
    elif opts["task"] == 2:
        dataset = ImageLabelAndLidarDataset(opts, datatype, transforms)

    dataloader = DataLoader(dataset, batch_size=opts[datatype]["batchsize"], shuffle=opts[datatype]["shuffle"], num_workers=opts[datatype]["num_workers"])

    return dataloader
