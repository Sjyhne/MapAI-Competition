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
from transforms import LidarAugComposer

from ensemble_model import EnsembleModel
import yaml

def main(args, pt_share_links):
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



    # models and configs for the ensemble
    model_name_list = [[]]
    model_cfg_list = [[]]

    max_ensemble_size = opts["models_per_ensemble"]
    for i, (pt_share_link, opt_share_link) in enumerate(pt_share_links):
        pt_id = pt_share_link.split("/")[-2]
        opt_id = opt_share_link.split("/")[-2]

        # Download trained model
        url_to_pt = f"https://drive.google.com/uc?id={pt_id}"
        url_to_opt = f"https://drive.google.com/uc?id={opt_id}"
        model_checkpoint = f"task1_pt{i + 1}.pt"
        model_cfg = f"task1_pt{i + 1}.yaml"

        gdown.download(url_to_pt, model_checkpoint, quiet=False)
        gdown.download(url_to_opt, model_cfg, quiet=False)

        if len(model_cfg_list[-1]) < max_ensemble_size:
            model_name_list[-1].append(model_checkpoint)
            model_cfg_list[-1].append(model_cfg)
            continue

        model_name_list.append([model_checkpoint])
        model_cfg_list.append([model_cfg])

    #########################################################################
    ###
    # Create needed directories for data
    ###
    #########################################################################
    task_path = pathlib.Path(args.submission_path).joinpath(f"task_{opts['task']}")
    temp_path = pathlib.Path(args.submission_path).joinpath(f"temp")
    
    opts_file = task_path.joinpath("opts.yaml")
    predictions_path = task_path.joinpath("predictions")
    if task_path.exists():
        shutil.rmtree(task_path.absolute())
    if temp_path.exists():
        shutil.rmtree(temp_path.absolute())
    
    temp_path.mkdir(exist_ok=True, parents=True)
    predictions_path.mkdir(exist_ok=True, parents=True)


    #########################################################################
    ###
    # Load Data
    ###
    #########################################################################

    target_size = (500, 500) # for resizing the predictions
    dataloader = create_dataloader(opts, opts["data_type"])
    print(dataloader)

    lidar_augs = LidarAugComposer(opts)
    _, lidar_valid = lidar_augs.get_transforms()

    device = opts["device"]
    for i, (configs, model_names) in enumerate(zip(model_cfg_list, model_name_list)):
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

            if config["imagesize"] != opts["imagesize"]:
                opts["imagesize"] = config["imagesize"]
                print(f"Using image resolution: {opts['imagesize']} * {opts['imagesize']}")

        model = EnsembleModel(models, target_size=target_size)
        model = model.to(device)
        model.eval()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), miniters=int(len(dataloader)/100), desc="Inference", leave=False)

        del models
        for idx, (image, label, filename) in pbar:
            # Split filename and extension
            filename_base, file_extension = os.path.splitext(filename[0])

            # Send image and label to device (eg., cuda)
            image = image.to(device)

            if opts["task"] == 2:
                image, lidar = torch.split(image, [3, 1], dim=1)
                lidar = lidar_valid(lidar.numpy())

                image = torch.cat([image, torch.tensor(lidar, dtype=image.dtype)], dim=1)


            # Perform model prediction
            prediction = model(image)["result"]

            # routput = model(torch.rot90(image, dims=[2, 3]))["result"]
            # routput = torch.rot90(routput, k=-1, dims=[2, 3])
            # prediction = (prediction + routput) / 2

            if opts["device"] == "cpu":
                prediction = prediction.squeeze().detach().numpy()
            else:
                prediction = prediction.squeeze().detach().cpu().numpy()
            

            assert prediction.shape == target_size, f"Prediction and label shape is not same, pls fix [{prediction.shape} - {target_size}]"

            if len(model_cfg_list) > 1:
                #Load and save temp prediction
                temp_pred_path = temp_path.joinpath(f"{filename_base}.npy")
                if i != 0:
                    prediction += np.load(str(temp_pred_path))
                if i < len(model_cfg_list) - 1:
                    np.save(str(temp_pred_path), prediction)

            if i == len(model_cfg_list) - 1:
                # Save final prediction
                if opts["device"] == "cpu":
                    image = image.squeeze().detach().numpy()[:3, :, :].transpose(1, 2, 0)
                else:
                    image = image.squeeze().cpu().detach().numpy()[:3, :, :].transpose(1, 2, 0)

                
                prediction = np.rint(prediction / len(model_cfg_list)).astype(np.uint8)
                if opts["post_process_preds"]:
                    prediction = post_process_mask(prediction)
                
                prediction_visual = np.copy(prediction)

                for idx, value in enumerate(opts["classes"]):
                    prediction_visual[prediction_visual == idx] = opts["class_to_color"][value]

                label = label.squeeze().detach().numpy()

                label = np.uint8(label)

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
        print()
        del model
    # Dump file configuration
    yaml.dump(opts, open(opts_file, "w"), Dumper=yaml.Dumper)
    shutil.rmtree(temp_path.absolute())
