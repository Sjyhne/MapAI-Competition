import cv2
import glob
import numpy as np
from tqdm import tqdm

batch_size = 64

if __name__ == "__main__":
    image_folder = r"./../../data/"
    image_paths = glob.glob(image_folder +"train/lidar/*.tif") + glob.glob(image_folder +"validation/lidar/*.tif")

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for i in tqdm(range(0, len(image_paths) - len(image_paths) % batch_size, batch_size)):
        paths = image_paths[i:i+batch_size]
        images = np.expand_dims(np.stack([cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in paths]).astype(int), axis=-1)
        print(images.shape)
        channels_sum += np.mean(images, axis=(0,1,2)) # mean channel values in the batch
        channels_squared_sum += np.mean(images**2, axis=(0,1,2)) # mean channel values of the images squared for the std calculation
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    print(mean) # [106.43160638 113.77681609 112.80221937, 1.75619225]
    print(std) # [40.47940348 40.96728925 46.88399016, 3.74813656]]