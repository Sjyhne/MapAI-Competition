#!/usr/bin/env python3

import glob
import os
import cv2

OUTPUT_DIR = "./mapai/"

TARGET_SIZE = 500
STRIDE = TARGET_SIZE // 3

IMGS_DIR = "images"
MASKS_DIR = "masks"
LIDAR_DIR = "lidar"

for split in ["train", "validation"]:
    img_paths = glob.glob(os.path.join(f"./../../data/big_tiles/{split}/{IMGS_DIR}", "*.tif"))
    mask_paths = glob.glob(os.path.join(f"./../../data/big_tiles/{split}/{MASKS_DIR}", "*.tif"))
    lidar_paths = glob.glob(os.path.join(f"./../../data/big_tiles/{split}/{LIDAR_DIR}", "*.tif"))

    img_paths.sort()
    mask_paths.sort()
    lidar_paths.sort()

    print(len(img_paths), len(lidar_paths), len(mask_paths))

    for i in range(len(img_paths)):
        img_path = img_paths[i]
        mask_path = mask_paths[i]
        lidar_path = lidar_paths[i]

        img_filename = os.path.splitext(os.path.basename(img_path))[0]
        mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
        lidar_filename = os.path.splitext(os.path.basename(lidar_path))[0]

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)  
        lidar = cv2.imread(lidar_path, cv2.IMREAD_UNCHANGED)

        assert img_filename == mask_filename and img.shape[:2] == mask.shape[:2]

        k = 0
        for y in range(0, img.shape[0], STRIDE):
            for x in range(0, img.shape[1], STRIDE):
                img_tile = img[y:y + TARGET_SIZE, x:x + TARGET_SIZE]
                mask_tile = mask[y:y + TARGET_SIZE, x:x + TARGET_SIZE]
                lidar_tile = lidar[y:y + TARGET_SIZE, x:x + TARGET_SIZE]

                if img_tile.shape[0] == TARGET_SIZE and img_tile.shape[1] == TARGET_SIZE:
                    out_img_path = os.path.join(OUTPUT_DIR + split, "images/{}_{}_{}.tif".format(img_filename, y // STRIDE, x // STRIDE))
                    cv2.imwrite(out_img_path, img_tile)

                    out_mask_path = os.path.join(OUTPUT_DIR + split, "masks/{}_{}_{}.tif".format(mask_filename, y // STRIDE, x // STRIDE))
                    cv2.imwrite(out_mask_path, mask_tile)

                    out_lidar_path = os.path.join(OUTPUT_DIR + split, "lidar/{}_{}_{}.tif".format(lidar_filename, y // STRIDE, x // STRIDE))
                    cv2.imwrite(out_lidar_path, lidar_tile)
                k += 1

        print("Processed {} {}/{}".format(img_filename, i + 1, len(img_paths)))
