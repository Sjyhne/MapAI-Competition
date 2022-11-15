import numpy as np
import glob
import cv2
from tqdm import tqdm
lidar = glob.glob(r".\..\..\data\overlapping_data\validation\lidar\*.tif")


min_all = float("inf")
max_all = 0

min_file = None
max_file = None

not_zero = []

for path in tqdm(lidar):
    gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    minimum = np.min(gt)
    if minimum < 0:
        not_zero.append(minimum)
    if minimum < min_all:
        min_all = minimum
        min_file = path
    if np.max(gt) > max_all:
        max_all = np.max(gt)
        max_file = path


print(min_all, max_all, min_file, max_file)
print(len(not_zero))