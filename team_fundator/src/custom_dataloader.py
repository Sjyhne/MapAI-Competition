import pathlib
from typing import Callable, Optional

from torch.utils.data import Dataset, DataLoader
import os
import torch
import cv2 as cv
import numpy as np
import pickle

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

    image = image.astype(np.float32) / 255.0
    #image = torch.permute(image, (2, 0, 1)) done by albummentations

    return image


def load_label(labelpath: str, size: tuple) -> torch.tensor:
    label = cv.imread(labelpath, cv.IMREAD_GRAYSCALE)
    label[label == 255] = 1
    label = cv.resize(label, size)

    label = label.astype(np.uint8)

    return label


def load_lidar(lidarpath: str, size: tuple) -> torch.tensor:
    lidar = cv.imread(lidarpath, cv.IMREAD_UNCHANGED)
    lidar = cv.resize(lidar, size)

    lidar = lidar.astype(np.float32)

    return lidar

def load_ensemble_pred(picklepath: str):
    # loads pickle files to train a metaensemble on stored predictions with shape N * H * W
    handle = open(picklepath, 'rb')
    data = pickle.load(handle)
    data = np.vstack(data)
    return data

class ImageAndLabelDataset(Dataset):

    def __init__(self,
                 opts: dict,
                 datatype: str = "validation",
                 transforms: Optional[Callable]=None,
                 aux_head_labels: bool=False,
                 use_lidar_in_mask: bool=False,
                 edge_only: bool=False):

        self.opts = opts
        self.aux_head_labels = aux_head_labels # wheter to add a label for an auxilliary classification head
        self.use_lidar_in_mask = use_lidar_in_mask # whether to use lidar to create a third class 
        self.ratio = self.opts[datatype]["data_ratio"]
        self.edge_only = edge_only

        root = opts["data_dirs"]["root"]
        folder = opts["data_dirs"][datatype]
        mask_dir = opts['data_dirs']['masks'] if datatype == "validation" or "masks_train" not in opts['data_dirs'] else opts['data_dirs']['masks_train'] 

        self.image_paths = sorted(pathlib.Path(f"{root}/{folder}/{opts['data_dirs']['images']}").glob("*.tif"))
        self.mask_paths = sorted(pathlib.Path(f"{root}/{folder}/{mask_dir}").glob("*.tif"))     

        self.lidar_paths = None
        if self.use_lidar_in_mask:
            self.lidar_paths = sorted(pathlib.Path(f"{root}/{folder}/{opts['data_dirs']['lidar']}").glob("*.tif"))
            assert len(self.image_paths)  == len(self.lidar_paths) 

        self.image_size = (opts["imagesize"], opts["imagesize"])
        self.label_size = self.image_size if datatype == "train" else (500, 500) # validate on the mask resolution sued in the competition
        assert len(self.image_paths)  == len(self.mask_paths) 
        print()

        print(
            f"Using number of images in {datatype}dataset: {int(len(self.image_paths) * self.ratio)}/{len(self.image_paths) }")
        self.transform = transforms
    
    def __len__(self):
        return int(len(self.image_paths) * self.ratio)

    def __getitem__(self, idx):

        imagefilepath = self.image_paths[idx].as_posix()
        labelfilepath = self.mask_paths[idx].as_posix()

        if self.use_lidar_in_mask:
            lidarfilepath = self.lidar_paths[idx].as_posix()
            assert imagefilepath.split("/")[-1] == lidarfilepath.split("/")[
                -1], f"imagefilename and labelfilename does not match; {imagefilepath.split('/')[-1]} != {lidarfilepath.split('/')[-1]}"
            lidar = load_lidar(lidarfilepath, self.label_size)

        assert imagefilepath.split("/")[-1] == labelfilepath.split("/")[
            -1], f"imagefilename and labelfilename does not match; {imagefilepath.split('/')[-1]} != {labelfilepath.split('/')[-1]}"

        filename = imagefilepath.split("/")[-1]

        image = load_image(imagefilepath, self.image_size)
        label = load_label(labelfilepath, self.label_size)
        
        if self.edge_only:
            label[label == 3.0] = 0

        # lidar == 0 indicates regions where a DTM and DSM are equal, therefore we give the option to add scuh regions as a new class
        if self.use_lidar_in_mask:
            condition = np.logical_and(lidar == 0.0, label != 1)
            label[condition] = 2

        sample = dict(
            id=filename,
            image=image,
            mask=label,
        )
        
        if self.transform is not None:
            sample = self.transform(**sample)
        else:
            sample["image"] = sample["image"].transpose(2, 0, 1)

        if self.aux_head_labels:
            # the auxilliary head takes binary labels which represent the prescence of buildings (1) in the GT mask
            sample["aux_label"] = np.expand_dims(np.any(label == 1.0), 0).astype(np.float32)
        sample["mask"] = np.expand_dims(sample["mask"], 0)
        return sample
    
    def set_transform(self, transform):
        self.transform = transform


class ImageLabelAndLidarDataset(Dataset):

    def __init__(self,
                 opts: dict,
                 datatype: str = "validation",
                 transform: Optional[Callable]=None,
                 lidar_transform: Optional[Callable]=None,
                 aux_head_labels: bool=False,
                 use_lidar_in_mask: bool=False,
                 lidar_only: bool=False,
                 edge_only: bool=False):

        self.opts = opts
        self.transform = transform
        self.lidar_transform = lidar_transform
        self.ratio = self.opts[datatype]["data_ratio"]
        self.aux_head_labels = aux_head_labels
        self.lidar_only = lidar_only
        self.use_lidar_in_mask = use_lidar_in_mask
        self.edge_only = edge_only

        root = opts["data_dirs"]["root"]
        folder = opts["data_dirs"][datatype]
        mask_dir = opts['data_dirs']['masks'] if datatype == "validation" or "masks_train" not in opts['data_dirs'] else opts['data_dirs']['masks_train'] 
        self.image_paths = sorted(pathlib.Path(f"{root}/{folder}/{opts['data_dirs']['images']}").glob("*.tif"))
        self.mask_paths = sorted(pathlib.Path(f"{root}/{folder}/{mask_dir}").glob("*.tif"))
        self.lidar_paths = sorted(pathlib.Path(f"{root}/{folder}/{opts['data_dirs']['lidar']}").glob("*.tif"))


        self.image_size = (opts["imagesize"], opts["imagesize"])
        self.label_size = self.image_size if datatype == "train" else (500, 500)

        assert len(self.image_paths)  == len(self.mask_paths) 
        assert len(self.image_paths)  == len(self.lidar_paths) 
        print(
            f"Using number of images in {datatype}dataset: {int(len(self.image_paths) * self.ratio)}/{len(self.image_paths) }")

    def __len__(self):
        return int(len(self.image_paths) * self.ratio)

    def __getitem__(self, idx):
        imagefilepath = self.image_paths[idx].as_posix()
        labelfilepath = self.mask_paths[idx].as_posix()
        lidarfilepath = self.lidar_paths[idx].as_posix()

        assert imagefilepath.split("/")[-1] == labelfilepath.split("/")[
            -1], f"imagefilename and labelfilename does not match; {imagefilepath.split('/')[-1]} != {labelfilepath.split('/')[-1]}"
        assert imagefilepath.split("/")[-1] == lidarfilepath.split("/")[
            -1], f"imagefilename and labelfilename does not match; {imagefilepath.split('/')[-1]} != {labelfilepath.split('/')[-1]}"

        filename = imagefilepath.split("/")[-1]

        image = None
        if not self.lidar_only:
            image = load_image(imagefilepath, self.image_size)
        label = load_label(labelfilepath, self.label_size)
        lidar = load_lidar(lidarfilepath, self.image_size)

        if self.edge_only:
            label[label == 3.0] = 0

        if self.use_lidar_in_mask:
            condition = np.logical_and(lidar == 0.0, label != 1)
            label[condition] = 2

        if self.transform is not None and not self.lidar_only:
            aug_sample = self.transform(image=image,  masks=[label, lidar]) # apply lidar augmentations as if it is a mask
            
            label, lidar = aug_sample['masks']
            image = aug_sample['image']
            assert image.dtype == lidar.dtype
        elif self.lidar_only:
            aug_sample = self.transform(image=lidar,  mask=label) # apply lidar augmentations as if it is an image
            label= aug_sample['mask']
            lidar = aug_sample['image']
        else:
            image = image.transpose(2, 0, 1)


        # Concatenate lidar and image data
        lidar = self.lidar_transform(lidar)
        if not self.lidar_only:
            lidar = np.expand_dims(lidar, 0)
            image = np.concatenate((image, lidar), axis=0)
        else:
            image = np.expand_dims(lidar, 0) # Use lidar as input, instead of image

        sample = dict(
            id=filename,
            image=image,
            mask=np.expand_dims(label, 0),
        )

        if self.aux_head_labels:
            sample["aux_label"] = np.expand_dims(np.any(label == 1.0), 0).astype(np.float32)
        # image2 = image.transpose(1, 2, 0)[:, :, :3].astype(np.float32) * 255
        # print(np.max(image2), np.max(label), np.max(lidar))
        # print(image.shape)
        # cv.imwrite("datatest/image.png", image2.astype(np.uint8))
        # cv.imwrite("datatest/lidar.tif", lidar.transpose(1, 2, 0).astype(np.float32))
        # cv.imwrite("datatest/label.tif", np.expand_dims(label, -1).astype(np.float32))
        # exit()
        return sample
    
    def set_transform(self, transform):
        self.transform = transform


class EnsembleDataset(Dataset):

    def __init__(self,
                 opts: dict,
                 datatype: str = "validation",
                 transforms: Optional[Callable]=None,
                 aux_head_labels: bool=False,
                 use_lidar_in_mask: bool=False):

        self.opts = opts
        self.datatype = datatype
        self.aux_head_labels = aux_head_labels # wheater to add a binary building / no building label for models with an auxilliary classification head
        self.use_lidar_in_mask = use_lidar_in_mask # whether to add a lidar-related class in the labels
        self.ratio = self.opts[datatype]["data_ratio"]

        root = opts["data_dirs"]["root"]
        base_root = opts["data_dirs"]["base_root"]
        folder = opts["data_dirs"][datatype]
        
        mask_dir = opts['data_dirs']['masks'] if datatype == "validation" or "masks_train" not in opts['data_dirs'] else opts['data_dirs']['masks_train'] 

        self.pickle_paths = sorted(pathlib.Path(f"{base_root}/ensembles/task{opts['ensemble_task']}/{opts['ensemble_name']}/{datatype}").glob("*.pickle"))
        mask_root = pathlib.Path(f"{root}/{folder}/{mask_dir}")
        self.mask_paths = sorted([mask_root / pickle_filename.with_suffix(".tif").name for pickle_filename in self.pickle_paths])     

        self.lidar_paths = None
        if self.use_lidar_in_mask:
            lidar_root = f"{root}/{folder}/{opts['data_dirs']['lidar']}"

            self.lidar_paths = sorted([lidar_root / pickle_filename.stem + ".tif" for pickle_filename in self.pickle_paths])   
            assert len(self.pickle_paths)  == len(self.lidar_paths) 

        self.label_size = (opts["imagesize"], opts["imagesize"]) if datatype == "train" else (500, 500)
        assert len(self.pickle_paths)  == len(self.mask_paths) 
        print()

        print(
            f"Using number of images in {datatype}dataset: {int(len(self.pickle_paths) * self.ratio)}/{len(self.pickle_paths) }")
        self.transform = transforms
    
    def __len__(self):
        return int(len(self.pickle_paths) * self.ratio)

    def __getitem__(self, idx):

        picklefilepath = self.pickle_paths[idx].as_posix()
        labelfilepath = self.mask_paths[idx].as_posix()

        if self.use_lidar_in_mask:
            lidarfilepath = self.lidar_paths[idx].as_posix()
            assert picklefilepath.split("/")[-1][:-7] == lidarfilepath.split("/")[
                -1][:-4], f"imagefilename and labelfilename does not match; {picklefilepath.split('/')[-1]} != {lidarfilepath.split('/')[-1]}"
            lidar = load_lidar(lidarfilepath, self.label_size)

        assert picklefilepath.split("/")[-1][:-7] == labelfilepath.split("/")[
            -1][:-4], f"imagefilename and labelfilename does not match; {picklefilepath.split('/')[-1]} != {labelfilepath.split('/')[-1]}"

        filename = picklefilepath.split("/")[-1]

        image = load_ensemble_pred(picklefilepath)
        label = load_label(labelfilepath, self.label_size)
        
        if self.use_lidar_in_mask:
            label[lidar == 0.0] = 2

        sample = dict(
            id=filename,
            image=image,
            mask=label,
        )


        if self.transform is not None and self.datatype == "train":
            sample = self.transform(**sample)

        if self.aux_head_labels:
            sample["aux_label"] = np.expand_dims(np.any(label == 1.0), 0).astype(np.float32)
        sample["mask"] = np.expand_dims(sample["mask"], 0)
        return sample
    
    def set_transform(self, transform):
        self.transform = transform

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


def create_dataloader(opts: dict, datatype: str = "test", transforms=None, aux_head_labels=False) -> DataLoader:
    image_transforms, lidar_transform = transforms
    use_lidar_in_mask = datatype == "train" and opts.get("use_lidar_in_mask", False)
    edge_only = opts["dataset"] == "mapai_edge"
    if opts["task"] == 1:
        dataset = ImageAndLabelDataset(opts, datatype, image_transforms, aux_head_labels, use_lidar_in_mask, edge_only=edge_only)
    elif opts["task"] == 4:
        dataset = EnsembleDataset(opts, datatype, image_transforms, aux_head_labels, use_lidar_in_mask)
    else:
        dataset = ImageLabelAndLidarDataset(opts, datatype, image_transforms, lidar_transform,  aux_head_labels, use_lidar_in_mask, lidar_only=opts["task"] == 3, edge_only=edge_only)

    dataloader = DataLoader(dataset, batch_size=opts[datatype]["batchsize"], shuffle=opts[datatype]["shuffle"], num_workers=opts[datatype]["num_workers"], drop_last=True)

    return dataloader
