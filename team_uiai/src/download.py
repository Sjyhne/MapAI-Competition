from competition_toolkit.dataloader import download_dataset
data_type = "train" # or "validation"
task = 2 # or 2, but does not matter for train and validation data types
# paths is a list of dicts, where each dict is a data sample, 
# and each dict have "image", "lidar", and "mask" keys
# with the path to the corresponding file
dataset = download_dataset(data_type, task, get_dataset=True)

print(dataset)
