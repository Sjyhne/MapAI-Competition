"""
@File           : check_data.py
@Author         : Gefei Kong
@Time:          : 12.10.2022 17:47
------------------------------------------------------------------------------------------------------------------------
@Description    : as below

"""

import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

image_file_path = '../../data/train/images/6051_689_0.tif'
lidar_file_path = '../../data/train/lidar/6051_689_0.tif'
mask_file_path = '../../data/train/masks/6051_689_0.tif'

image = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)
lidar = cv2.imread(lidar_file_path, cv2.IMREAD_UNCHANGED)
mask  = cv2.imread(mask_file_path, cv2.IMREAD_UNCHANGED)

print(image.shape, lidar.shape, mask.shape)
print(np.max(lidar), np.min(lidar))

fig, ax = plt.subplots(3,1, figsize=(6,12))
ax[0].imshow(image)
ax[1].imshow(lidar)
ax[2].imshow(mask)
# plt.imshow(lidar)
plt.show()
