## define miou and biou
import cv2
import numpy as np
import argparse
import glob
import json
from pathlib import Path

from competition_toolkit.dataloader import create_dataloader, download_dataset
from competition_toolkit.eval_functions import iou, biou

def class_wise(arr: np.array, c: int) -> np.array:
    return arr == c

def iou(prediction: np.array, target: np.array) -> float:

    miou = []

    for c in range(2):

        pred = class_wise(prediction, c)
        tar = class_wise(target, c)

        if pred.dtype != bool:
            pred = np.asarray(pred, dtype=bool)

        if tar.dtype != bool:
            tar = np.asarray(tar, dtype=bool)

        overlap = pred * tar # Logical AND
        union = pred + tar # Logical OR

        if union.sum() != 0 and overlap.sum() != 0:
            iou = (float(overlap.sum()) / float(union.sum()))
        else:
            iou = 0

        if c in target:
            miou.append(iou)

    return np.asarray(miou).mean()

# General util function to get the boundary of a binary mask.
def _mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def biou(dt, gt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """

    mboundary_iou = []

    for c in range(2):

        target = class_wise(gt, c)
        if not np.any(target):
            continue

        prediction = class_wise(dt, c)

        gt_boundary = _mask_to_boundary(target.astype(np.uint8), dilation_ratio)
        dt_boundary = _mask_to_boundary(prediction.astype(np.uint8), dilation_ratio)
        intersection = ((gt_boundary * dt_boundary) > 0).sum()
        union = ((gt_boundary + dt_boundary) > 0).sum()
        if union == 0 or intersection == 0:
            boundary_iou = 0
        else:
            boundary_iou = (intersection / union)

        mboundary_iou.append(boundary_iou)

    return np.asarray(mboundary_iou).mean()

def calculate_score(preds: np.array, tars: np.array) -> dict:

    assert preds.shape == tars.shape, f"pred shape {preds.shape} does not match tar shape {tars.shape}"
    assert len(preds.shape) != 4, f"expected shape is (bs, ydim, xdim), but found {preds.shape}"
    assert type(preds) == np.ndarray, f"preds is a {type(preds)}, but should be numpy.ndarray"
    assert type(tars) == np.ndarray, f"tars is a {type(tars)}, but should be numpy.ndarray"
    assert type(preds[0][0][0]) == np.uint8, f"preds is not of type np.uint8, but {type(preds[0][0][0])}"
    assert type(tars[0][0][0]) == np.uint8, f"tars is not of type np.uint8, but {type(tars[0][0][0])}"

    bs = preds.shape[0]

    t_iou = 0
    t_biou = 0

    for i in range(bs):
        t_iou += iou(preds[i], tars[i])
        t_biou += biou(preds[i], tars[i])

    t_iou /= bs
    t_biou /= bs

    score = (t_iou + t_biou) / 2

    return {"score": score, "iou": t_iou, "biou": t_biou}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=1, help="Which task to evaluate")
    parser.add_argument("--data-ratio", type=float, default=1.0)
    parser.add_argument("--data-type", type=str, help="validation or test")
    parser.add_argument("--submission-path", type=str, required=True, help="path to participant")
    parser.add_argument("--team", type=str, default='Deepcrop', required=True)
    parser.add_argument("--gt-root", type=str, default="data/validation/masks/", required=True, help="Path to test_root mask")
    # gt_root = "/home/dmn774/data/Deep1/benchmarks/NORA_MapAI/data/validation/masks/"

    # img_root = "/home/dmn774/data/Deep1/benchmarks/NORA_MapAI/data/validation/images/"
    # model_predict_path = 'show_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom80k_aug5000/iter_41600_mask/*.tif'
    args = parser.parse_args()
    model_predict_path  = args.submission_path + "/*.tif"
    dataset_path = download_dataset(task=args.task, data_type=args.data_type)
    mask_path = dataset_path.joinpath("masks")  # TODO correct?
    gt_root = args.gt_root

    count = 0
    avg_iou = 0
    avg_biou = 0
    all_iou = []
    all_biou = []

    for line in glob.glob(model_predict_path):
        item = line.split("/")[-1]
        # for line in val_list:
        #     item = line.strip()
        count += 1
        # pre_file = line
        dim = (500, 500)
        pre_img = cv2.imread(line, cv2.IMREAD_GRAYSCALE)
        pre_img = cv2.resize(pre_img, dim, interpolation=cv2.INTER_AREA)

        pre_img[pre_img == 0] = 1
        pre_img[pre_img == 255] = 0

        gt_sample = gt_root + item
        # gt_sample = mask_path.joinpath(item)
        # import ipdb; ipdb.set_trace()
        label = cv2.imread(gt_sample, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, dim, interpolation=cv2.INTER_AREA)


        label[label == 255] = 1
        zero_ = np.zeros_like(label)
        combine_ = np.stack([pre_img, label, zero_], axis=-1) * 255
        tem_iou = iou(pre_img, label)
        tem_biou = biou(pre_img, label)
        all_iou.append(tem_iou)
        all_biou.append(tem_biou)

    print(count)
    avg_iou = np.round(np.sum(all_iou) / count, 5)
    avg_biou = np.round(np.sum(all_biou) / count, 5)
    # print("after processing")
    print("avg_iou", avg_iou, "avg_biou", avg_biou)
    # a_iou = np.argsort(all_iou)
    # b_iou = np.argsort(all_biou)
