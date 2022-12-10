from competition_toolkit.dataloader import download_dataset

import datasets
data_type = "train" # or "validation"
task = 1 # or 2, but does not matter for train and validation data types
# paths is a list of dicts, where each dict is a data sample, 
# and each dict have "image", "lidar", and "mask" keys
# with the path to the corresponding file
paths = download_dataset(data_type, task)
# datasets.load_dataset("csv", )
data_sample = paths[0]

"""
data_sample = {
    "image": <path_to_img>,
    "lidar": <path_to_lidar>,
    "mask": <path_to_mask>
}
"""