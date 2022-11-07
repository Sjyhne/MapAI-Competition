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

from kornia.morphology import erosion, dilation

from competition_toolkit.dataloader import create_dataloader
from competition_toolkit.eval_functions import iou, biou

from utils import get_model

from ensemble_model import EnsembleModel, load_models_from_runs
from transforms import valid_transform, LidarAugComposer

import yaml

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
    pt_share_links = [
            (
            "https://drive.google.com/file/d/1AWduWlYKH0SdfMhZ0cT0Ymj9HBSGF18t/view?usp=share_link", # cp
            "https://drive.google.com/file/d/1PXa7AlWPIOAUZEmHQyXGahK2MRIFIRXG/view?usp=share_link" # opts
            ),
        ]

    model_names = []
    configs = []

    for i, (pt_share_link, opt_share_link) in enumerate(pt_share_links):
        pt_id = pt_share_link.split("/")[-2]
        opt_id = opt_share_link.split("/")[-2]

        # Download trained model ready for inference
        url_to_pt = f"https://drive.google.com/uc?id={pt_id}"
        url_to_opt = f"https://drive.google.com/uc?id={opt_id}"
        model_checkpoint = f"task1_pt{i + 1}.pt"
        model_cfg = f"task1_pt{i + 1}.yaml"

        gdown.download(url_to_pt, model_checkpoint, quiet=False)
        gdown.download(url_to_opt, model_cfg, quiet=False)

        model_names.append(model_checkpoint)
        configs.append(model_cfg)

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


    models = []
    for config, checkpoint in zip(configs, model_names):
        config =  yaml.load(open(config, "r"), yaml.Loader)
        model = get_model(config)
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device(opts["device"])))
        models.append(model)

    target_size = (500, 500) if opts["data_type"] == "test" else (opts["imagesize"], opts["imagesize"])

    model = EnsembleModel(models, target_size=target_size)
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

    for idx, (image, label, filename) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Inference",
                                              leave=False):
        # Split filename and extension
        filename_base, file_extension = os.path.splitext(filename[0])

        # Send image and label to device (eg., cuda)
        image = image.to(device)

        # Perform model prediction
        output = model(image[:, :3])["result"]
        output = torch.round(output)

        if opts["erode_val_preds"]:
            kernel = torch.ones(5, 5).to(device)
            output = erosion(output, kernel)
            output = dilation(output, kernel).squeeze()

        if opts["device"] == "cpu":
            #prediction = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze().detach().numpy()
            prediction = output.squeeze().detach().numpy()
        else:
            #prediction = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze().cpu().detach().numpy()
            prediction = output.squeeze().cpu().detach().numpy()
        # Postprocess prediction

        label = label.squeeze().detach().numpy()

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

    print("iou_score:", np.round(iou_scores.mean(), 5), "biou_score:", np.round(biou_scores.mean(), 5))

    # Dump file configuration
    yaml.dump(opts, open(opts_file, "w"), Dumper=yaml.Dumper)
