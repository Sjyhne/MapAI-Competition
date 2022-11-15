import pathlib

import cv2 as cv
from utils import post_process_mask
import argparse
from tqdm import tqdm

from pathlib import Path

from competition_toolkit.eval_functions import iou, biou



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=1, help="Which task to evaluate")

    args = parser.parse_args()

    dataset_path = Path('../../data/validation')
    mask_path = dataset_path.joinpath("masks")


    submission_path = Path('submission')
    task_path = submission_path.joinpath("task_" + str(args.task))
    prediction_path = task_path.joinpath("predictions")

    validation_files = pathlib.Path(mask_path).glob("**/*.tif")
    pred_files = {
        file.name: [file] for file in prediction_path.glob("**/*.tif")
    }

    for val_file in validation_files:
        if val_file.name in pred_files:
            pred_files[val_file.name].append(val_file)


    iou_scores = 0
    biou_scores = 0
    p_process = True

    for i, (sample_name, (pred_path, mask_path)) in tqdm(enumerate(pred_files.items()), total=len(pred_files)):

        pred = cv.imread(str(pred_path), cv.IMREAD_GRAYSCALE)
        mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

        if p_process:
            pred = post_process_mask(pred)

        iou_score = iou(pred, mask)
        biou_score = biou(pred, mask)

        iou_scores += iou_score
        biou_scores += biou_score

    iscore = iou_scores / len(pred_files)
    bscore = biou_scores / len(pred_files)
    score = (iscore + bscore) / 2
    print(iscore, bscore, score)
