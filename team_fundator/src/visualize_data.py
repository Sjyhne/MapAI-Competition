import argparse
import cv2


def visualize_image(image, subset="train", preds_folder=None):
    # Load base image, Lidar image, ground truth mask and optionally predicted mask
    base_image = cv2.imread(f"../data/{subset}/images/{image}.tif")
    lidar_image = cv2.imread(f"../data/{subset}/lidar/{image}.tif")
    gt_mask = cv2.imread(f"../data/{subset}/masks/{image}.tif")

    if preds_folder is not None:
        pred_mask = cv2.imread(f"{preds_folder}/{image}_classified.jpg")

    


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--")