import pathlib

from tqdm import tqdm
import torch
import numpy as np
import cv2 as cv
import yaml
import matplotlib.pyplot as plt
import gdown
import os
import shutil


from competition_toolkit.dataloader import create_dataloader

from utils import get_model, post_process_mask

from ensemble_model import EnsembleModel
from transforms import LidarAugComposer

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

    lidar_augs = LidarAugComposer(opts)
    _, lidar_valid = lidar_augs.get_transforms()

    models = []
    for config, checkpoint in zip(configs, model_names):
        config =  yaml.load(open(config, "r"), yaml.Loader)
        model = get_model(config)
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device(opts["device"])))
        models.append(model)

    target_size = (500, 500) 
    
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

    for idx, (image, label, filename) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Inference",
                                              leave=False):
        # Split filename and extension
        filename_base, file_extension = os.path.splitext(filename[0])

        image, lidar = torch.split(image, [3, 1], dim=1)
        lidar = lidar_valid(lidar.numpy())

        image = torch.cat([image, torch.tensor(lidar, dtype=image.dtype)], dim=1)

        # Send image and label to device (eg., cuda)
        image = image.to(device)

        # Perform model prediction
        output = model(image[:, :3])["result"]
        output = torch.round(output)

        if opts["device"] == "cpu":
            #prediction = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze().detach().numpy()
            prediction = output.squeeze().detach().numpy().astype(np.uint8)
        else:
            #prediction = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze().cpu().detach().numpy()
            prediction = output.squeeze().detach().cpu().numpy().astype(np.uint8)

        if opts["post_process_preds"]:
            prediction = post_process_mask(prediction)
        

        label = label.squeeze().detach().numpy()

        label = np.uint8(label)
        assert prediction.shape == target_size, f"Prediction and label shape is not same, pls fix [{prediction.shape} - {target_size}]"

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

    # Dump file configuration
    yaml.dump(opts, open(opts_file, "w"), Dumper=yaml.Dumper)
