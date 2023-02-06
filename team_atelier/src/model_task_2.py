import pathlib

from tqdm import tqdm
import torch
import torchvision
import numpy as np
import cv2 as cv
import yaml
import matplotlib.pyplot as plt
import gdown
import os
import shutil

from competition_toolkit.dataloader import create_dataloader
from competition_toolkit.eval_functions import iou, biou

from utils import create_model, predict


def main(args):
    #########################################################################
    ###
    # Load Model and its configuration
    ###
    #########################################################################
    with open(args.config, "r") as f:
        opts = yaml.load(f, Loader=yaml.Loader)
        opts = {**opts, **vars(args)}

    #########################################################################
    ###
    # Download Model Weights
    # Use a mirror that is publicly available. This example uses Google Drive
    ###
    #########################################################################

    onnx_share_link = "https://drive.google.com/file/d/1PqoTCypCxlhm_0PhCp3G2rtY1YVSXEod/view?usp=share_link"

    # Download trained model ready for inference
    model_checkpoint = "task2.onnx"

    gdown.download(onnx_share_link, model_checkpoint, quiet=False, fuzzy=True)

    #########################################################################
    ###
    # Create needed directories for data
    ###
    #########################################################################
    task_path = pathlib.Path(args.submission_path).joinpath(f"task_{opts['task']}")
    opts_file = task_path.joinpath("opts.yaml")
    predictions_path = task_path.joinpath("predictions")
    if task_path.exists():
        shutil.rmtree(task_path.absolute())
    predictions_path.mkdir(exist_ok=True, parents=True)

    #########################################################################
    ###
    # Setup Model
    ###
    #########################################################################
    # Adds 4 channels to the input layer instead of 3

    model = create_model("task2.onnx", cpu=args.device=="cpu")

    #########################################################################
    ###
    # Load Data
    ###
    #########################################################################

    dataloader = create_dataloader(opts, opts["data_type"])


    iou_scores = np.zeros((len(dataloader)))
    biou_scores = np.zeros((len(dataloader)))

    for idx, (image, label, filename) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Inference", leave=False):

        assert image.shape[0] == 1, "The inference can only handle batch of 1"

        # Split filename and extension
        filename_base, file_extension = os.path.splitext(filename[0])


        # Perform model prediction
        prediction = predict(image, [model,])

        # softmax is waste of resources, argmax(softmax(x)) == argmax(x)
        prediction = torch.argmax(prediction, dim=1).detach().cpu().squeeze().numpy()

        prediction = np.uint8(prediction)
        label = np.uint8(label.squeeze())

        assert prediction.shape == label.shape, f"Prediction and label shape is not same, pls fix [{prediction.shape} - {label.shape}]"

        # Predict score
        iou_score = iou(prediction, label)
        biou_score = biou(label, prediction)

        iou_scores[idx] = np.round(iou_score, 6)
        biou_scores[idx] = np.round(biou_score, 6)

        prediction_visual = np.copy(prediction)

        for idx, value in enumerate(opts["classes"]):
            prediction_visual[prediction_visual == idx] = opts["class_to_color"][value]

        if opts["device"] == "cpu":
            image = image.squeeze().detach().numpy()[:3, :, :].transpose(1, 2, 0)
        else:
            image = image.squeeze().cpu().detach().numpy()[:3, :, :].transpose(1, 2, 0)

        fig, ax = plt.subplots(1, 3)

        ax[0].set_title("Input (RGB)")
        ax[0].imshow(image)
        ax[1].set_title("Prediction")
        ax[1].imshow(prediction_visual)
        ax[2].set_title("Label")
        ax[2].imshow(label)

        # Save to file.
        predicted_sample_path_png = predictions_path.joinpath(f"{filename_base}.png")
        predicted_sample_path_tif = predictions_path.joinpath(filename[0])
        plt.savefig(str(predicted_sample_path_png))
        plt.close()
        cv.imwrite(str(predicted_sample_path_tif), prediction)

    print("iou_score:", np.round(iou_scores.mean(), 5), "biou_score:", np.round(biou_scores.mean(), 5))

    # Dump file configuration
    yaml.dump(opts, open(opts_file, "w"), Dumper=yaml.Dumper)
