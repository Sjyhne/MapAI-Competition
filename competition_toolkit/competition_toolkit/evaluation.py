import pathlib

import cv2 as cv
import numpy as np
import tempfile
import argparse

import json
from pathlib import Path

from eval_functions import iou, biou
from dataloader import download_dataset


# NOT USED
def calculate_score(preds: np.array, tars: np.array) -> dict:
    assert preds.shape == tars.shape, f"pred shape {preds.shape} does not match tar shape {tars.shape}"
    assert len(preds.shape) != 4, f"expected shape is (bs, ydim, xdim), but found {preds.shape}"
    assert type(preds) == np.ndarray, f"preds is a {type(preds)}, but should be numpy.ndarray"
    assert type(tars) == np.ndarray, f"tars is a {type(tars)}, but should be numpy.ndarray"
    assert type(preds[0][0][0]) == np.uint8, f"preds is not of type np.uint8, but {type(preds[0][0][0])}"
    assert type(tars[0][0][0]) == np.uint8, f"tars is not of ttype np.uint8, but {type(tars[0][0][0])}"

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
    parser.add_argument("--team", type=str, required=True)

    args = parser.parse_args()

    #############################################################
    # Download dataset DO NOT EDIT
    #############################################################
    dataset_path = download_dataset(task=args.task, data_type=args.data_type)
    mask_path = dataset_path.joinpath("masks")  # TODO correct?


    #base_path = Path(__file__).parent.parent
    submission_path = Path(args.submission_path)
    task_path = submission_path.joinpath("task_" + str(args.task))
    prediction_path = task_path.joinpath("predictions")
    prediction_path.mkdir(exist_ok=True, parents=True)

    validation_files = pathlib.Path(mask_path).glob("**/*.tif")

    pred_files = {
        file.name: [file] for file in prediction_path.glob("**/*.tif")
    }

    for val_file in validation_files:
        if val_file.name in pred_files:
            pred_files[val_file.name].append(val_file)

    iou_scores = 0
    biou_scores = 0
    for sample_name, (pred_path, mask_path) in pred_files.items():

        pred = cv.imread(str(pred_path), cv.IMREAD_GRAYSCALE)
        mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

        iou_score = iou(pred, mask)
        biou_score = biou(pred, mask)

        iou_scores += iou_score
        biou_scores += biou_score

    iscore = np.round(iou_scores / len(pred_files), 4)
    bscore = np.round(biou_scores / len(pred_files), 4)

    result_file_name = f"{args.team.replace('/', '')}_task_{args.task}.json"
    results_dir = pathlib.Path("/tmp/mapai-result-artifacts")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir.joinpath(result_file_name), "w+") as f:
        result_dict = {"iou": iscore, "biou": bscore}
        json.dump(result_dict, f)

    print(f.name)
