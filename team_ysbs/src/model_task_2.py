import os
import pathlib
import shutil

import cv2 as cv
import gdown
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from competition_toolkit.dataloader import create_dataloader
from competition_toolkit.eval_functions import iou, biou
from tqdm import tqdm

from models import load_model


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
    pt_share_link = "https://drive.google.com/file/d/1klwCVV2q53p3DLdEV8Lzv2S2vKUS2s6D/view?usp=sharing"
    pt_id = pt_share_link.split("/")[-2]

    # Download trained model ready for inference
    url_to_drive = f"https://drive.google.com/uc?id={pt_id}&confirm=t"
    model_checkpoint = "pretrained_task2.pt"

    gdown.download(url_to_drive, model_checkpoint, quiet=False)

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
    model, out = load_model(model_name='transunet', opts=opts)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    device = opts["device"]
    model = model.to(device)
    model.eval()

    #########################################################################
    ###
    # Load Data
    ###
    #########################################################################

    dataloader = create_dataloader(opts, opts["data_type"])
    print(dataloader)

    iou_scores = np.zeros((len(dataloader)))
    biou_scores = np.zeros((len(dataloader)))

    for idx, (image, label, filename) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Inference", leave=False):
        # Split filename and extension
        filename_base, file_extension = os.path.splitext(filename[0])

        # Send image and label to device (eg., cuda)
        image = image.to(device)
        label = label.to(device)

        # Perform model prediction
        if out != -1:
            prediction = model(image)[out]
        else:
            prediction = model(image)

        if opts["device"] == "cpu":
            prediction = torch.argmax(torch.softmax(prediction, dim=1), dim=1).squeeze().detach().numpy()
        else:
            prediction = torch.argmax(torch.softmax(prediction, dim=1), dim=1).squeeze().cpu().detach().numpy()

        # Postprocess prediction
        if opts["device"] == "cpu":
            label = label.squeeze().detach().numpy()
        else:
            label = label.squeeze().cpu().detach().numpy()
        prediction = np.uint8(prediction)
        label = np.uint8(label)
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

        resize_image = cv.imread(str(predicted_sample_path_tif), 1)
        resized_image_out = cv.resize(resize_image, (500, 500))
        cv.imwrite(str(predicted_sample_path_tif), resized_image_out)

    print("iou_score:", np.round(iou_scores.mean(), 5), "biou_score:", np.round(biou_scores.mean(), 5))

    # Dump file configuration
    yaml.dump(opts, open(opts_file, "w"), Dumper=yaml.Dumper)
