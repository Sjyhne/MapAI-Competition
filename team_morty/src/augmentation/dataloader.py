# this wil owerwrite the dataloader so it uses augmetation before creating tensors

import albumentations as A
import pathlib
from torch.utils.data import Dataset, DataLoader
from yaml import load, Loader
from datasets import load_dataset
import os
import torch
import cv2 as cv
import numpy as np
from augmentation.data_augmentation import get_transforms
from torch.utils.data import RandomSampler 
from random import choice


transform = get_transforms()

class DS(Dataset):
    def __len__(self):
        return 5
    def __getitem__(self, index):
        return torch.empty(1).fill_(index)


def load_image(imagepath: str, size: tuple) -> torch.tensor:
    image = cv.imread(imagepath, cv.IMREAD_COLOR)
    #image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return cv.resize(image, size)


def load_lidar(lidarpath: str, size: tuple) -> torch.tensor:
    lidar = cv.imread(lidarpath, cv.IMREAD_UNCHANGED)
    return cv.resize(lidar, size)


def load_label(labelpath: str, size: tuple) -> torch.tensor:
    label = cv.imread(labelpath, cv.IMREAD_GRAYSCALE)
    label[label == 255] = 1
    # label[label >= 128] = 1
    # label[label < 128] = 0
    return cv.resize(label, size)


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


def download_dataset(data_type: str, task: int, get_dataset: bool = False):
    if data_type == "test":
        paths = load_dataset("sjyhne/mapai_evaluation_data", split=f"task{str(task)}", use_auth_token=True)
    else:
        paths = load_dataset("sjyhne/mapai_training_data", split=data_type)

    if get_dataset:
        return paths

    single_path = pathlib.Path(paths[0]["image"]).parent.parent.absolute()
    return single_path

class Task_One_Augmented_dataset(Dataset):
    def __init__(self,
                opts: dict,
                datatype: str = "validation"):

        self.opts = opts

        #self.paths = download_dataset(data_type=datatype, task=opts["task"], get_dataset=True)
        # paths = []
        # for path, size_reduce, keep_original in DATASET_FOLDER_PATH:
        #     paths.extend(self.construct_psuedo_paths(path, size_reducer=size_reduce, keep_original=keep_original))

        #self.paths = paths#self.construct_psuedo_paths(size_reducer=SIZE_REDUCER)
        self.paths = self.draw_dataset_files(DATASET_FOLDERS, DATASET_ROOT_FOLDER, ORIGINAL_DATASET_FOLDER, MASKS_FOLDER_PATH)

        print(
            f"Using number of images in {datatype}dataset: {int(len(self.paths) * self.opts['data_ratio'])}/{len(self.paths)}")
    
    def preprocess_dataset_folders(self, dataset_folders, dataset_root):
        file_iterations_dict = {}
        for dataset_folder in dataset_folders:
            for file_name in os.listdir(os.path.join(dataset_root, dataset_folder)):
                file_name_base = "_".join(file_name.split("_")[:-1])
                if file_name_base not in file_iterations_dict:
                    file_iterations_dict[file_name_base] = []
                file_iterations_dict[file_name_base].append(os.path.join(dataset_folder, file_name))
        return file_iterations_dict

    def draw_dataset_files(self, dataset_folders, dataset_root, original_dataset_folder, original_mask_folder):
        preprocessed_files = self.preprocess_dataset_folders(dataset_folders, dataset_root)
        selected_paths = []
        for original_file in os.listdir(original_dataset_folder):
            original_file_base = original_file.split(".")[0]
            file_iterations = preprocessed_files[original_file_base]
            drawn_file = choice(file_iterations)
            selected_paths.append({"image": os.path.join(dataset_root,drawn_file), "mask": os.path.join(original_mask_folder,original_file)})
        return selected_paths

    def get_original_file_name(self, augmented_file_name: str):
        original_name = augmented_file_name.split("_")[:-1]
        original_name = "_".join(original_name) + ".tif"
        return original_name
    
    def get_file_iteration_number(self, filename: str):
        return int(filename.split("_")[-1].split(".")[0])

    def construct_randomized_paths(self, data_path):
        file_names = os.listdir(f"{data_path}")
        paths = []
        for file_name in file_names:
            if file_name == ".ipynb_checkpoints":
                continue
            paths.append({"image": f"{data_path}/{file_name}", "mask": f"{MASKS_FOLDER_PATH}/{file_name}"})
        
        return paths

    def construct_psuedo_paths(self, data_path, size_reducer = -1, keep_original = True):
        file_names = os.listdir(f"{data_path}")
        paths = []
        for file_name in file_names:
            if file_name == ".ipynb_checkpoints":
                continue
            iteration_number = self.get_file_iteration_number(file_name)
            if size_reducer != -1 and iteration_number > size_reducer:
                continue
            if iteration_number == 0 and not keep_original:
                continue
            original_file_name = self.get_original_file_name(file_name)
            paths.append({"image": f"{data_path}/{file_name}", "mask": f"{MASKS_FOLDER_PATH}/{original_file_name}"})
        
        return paths

    def __len__(self):
        return int(len(self.paths) * self.opts["data_ratio"])

    def __getitem__(self, idx):
        pathdict = self.paths[idx]

        imagefilepath = pathdict["image"]
        labelfilepath = pathdict["mask"]

        filename = imagefilepath.split("/")[-1]

        image = load_image(imagefilepath, (self.opts["imagesize"], self.opts["imagesize"]))
        label = load_label(labelfilepath, (self.opts["imagesize"], self.opts["imagesize"]))
        t = transform(image=image, mask=label)
        image = t["image"]
        image = torch.tensor(image.astype(np.uint8), device="cuda") / 255
        image = torch.permute(image, (2, 0, 1))
        label = t["mask"]
        label = torch.tensor(label.astype(np.uint8), device="cuda").long()
        assert image.shape[1:] == label.shape[
                                  :2], f"image and label shape not the same; {image.shape[1:]} != {label.shape[:2]}"

        return image, label, filename


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
        t = transform(image=image, mask=label)
        image = t["image"]
        image = torch.tensor(image.astype(np.uint8), device="cuda") / 255
        image = torch.permute(image, (2, 0, 1))
        label = t["mask"]
        label = torch.tensor(label.astype(np.uint8), device="cuda").long()
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
        lidar = load_lidar(lidarfilepath, (self.opts["imagesize"], self.opts["imagesize"]))

        
        t = transform(image=image, image1=lidar, mask=label)
        lidar = t["image1"]
        lidar = torch.tensor(lidar.astype(np.float), device="cuda").float()
        image = t["image"]
        image = torch.tensor(image.astype(np.uint8), device="cuda") / 255
        image = torch.permute(image, (2, 0, 1))
        label = t["mask"]
        label = torch.tensor(label.astype(np.uint8), device="cuda").long()

  

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
    if opts["task"] == 1 and datatype == "train":
        dataset = Task_One_Augmented_dataset(opts, datatype)    
    elif opts["task"] == 1:
        dataset = ImageAndLabelDataset(opts, datatype)
    # if opts["task"] == 1:
    #     dataset = ImageAndLabelDataset(opts, datatype)
    elif opts["task"] == 2:
        dataset = ImageLabelAndLidarDataset(opts, datatype)

    ds = DS()
    sampler = RandomSampler(ds, replacement=True, num_samples=3000)
    dataloader = DataLoader(dataset, batch_size=opts[f"task{opts['task']}"]["batchsize"], shuffle=opts[f"task{opts['task']}"]["shuffle"])#, sampler=sampler)

    return dataloader

import os
import re
ORIGINAL_DATASET_FOLDER = "D:/data/train/images"

if __name__ == "__main__":
    print(os.listdir(ORIGINAL_DATASET_FOLDER))
    #opts = load(open("config/massachusetts.yaml"), Loader=Loader)
    #dataset = Task_One_Augmented_dataset(opts, "train")
    # testloader = create_dataloader(opts, "test")

    # for batch in testloader:
    #     image, label, filename = batch

    #     print("image.shape:", image.shape)
    #     print("label.shape:", label.shape)
    #     print("filename:", filename)

    #     exit()
