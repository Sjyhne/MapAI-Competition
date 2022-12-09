from competition_toolkit.dataloader import download_dataset
task = 1 # or 2, but does not matter for train and validation data types
# paths is a list of dicts, where each dict is a data sample, 
# and each dict have "image", "lidar", and "mask" keys
# with the path to the corresponding file
paths = download_dataset("train", task)