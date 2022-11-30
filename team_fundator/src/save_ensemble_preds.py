import argparse

import pathlib
from tqdm import tqdm
import torch
import numpy as np
import yaml
import gdown
import os
import shutil


from competition_toolkit.dataloader import create_dataloader

from utils import get_model
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
    # Create needed directories for data
    ###
    #########################################################################
    task_path = pathlib.Path(args.submission_path).joinpath(f"task_{opts['task']}")
    temp_path = pathlib.Path(args.submission_path).joinpath(f"temp")

    # if task_path.exists():
    #     shutil.rmtree(task_path.absolute())
    # if temp_path.exists():
    #     shutil.rmtree(temp_path.absolute())
    
    temp_path.mkdir(exist_ok=True, parents=True)
    task_path.mkdir(exist_ok=True, parents=True)

    # models and configs for the ensemble
    model_name_list = [[]]
    model_cfg_list = [[]]

    max_ensemble_size = 1
    for i, (pt_share_link, opt_share_link) in enumerate(pt_share_links):
        pt_id = pt_share_link.split("/")[-2]
        opt_id = opt_share_link.split("/")[-2]

        # Download trained model
        url_to_pt = f"https://drive.google.com/uc?id={pt_id}"
        url_to_opt = f"https://drive.google.com/uc?id={opt_id}"
        model_checkpoint = temp_path.joinpath(f"task{opts['task']}_pt{i + 1}.pt").absolute()
        model_cfg = temp_path.joinpath(f"task{opts['task']}_pt{i + 1}.yaml").absolute()

        # if i == 5:
        #     gdown.download(url_to_pt, str(model_checkpoint), quiet=False)
        #     gdown.download(url_to_opt, str(model_cfg), quiet=False)

        if len(model_cfg_list[-1]) < max_ensemble_size:
            model_name_list[-1].append(model_checkpoint)
            model_cfg_list[-1].append(model_cfg)
            continue

        model_name_list.append([model_checkpoint])
        model_cfg_list.append([model_cfg])


    target_size = (500, 500) # for resizing the predictions
    dataloader = create_dataloader(opts, opts["data_type"])

    lidar_augs = LidarAugComposer(opts)
    _, lidar_valid = lidar_augs.get_transforms()

    device = opts["device"]
    for i, (configs, model_names) in enumerate(zip(model_cfg_list, model_name_list)):
        if i != 7:
            continue
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
        pbar = tqdm(dataloader, miniters=int(len(dataloader)/100), desc=f"Inference - Iter {i + 1}/{len(model_cfg_list)}")

        del models
        for image, label, filename in pbar:
            # Split filename and extension
            filename_base, file_extension = os.path.splitext(filename[0])


            if opts["task"] == 2:
                image, lidar = torch.split(image, [3, 1], dim=1)
                lidar = lidar_valid(lidar.numpy())

                image = torch.cat([image, torch.tensor(lidar, dtype=image.dtype)], dim=1)

            # Send image and label to device (eg., cuda)
            image = image.to(device)

            # Perform model prediction
            prediction = model(image)["result"]
            if opts["task"] == 2:
                routput = model(torch.rot90(image, dims=[2, 3]))["result"]

                routput = torch.rot90(routput, k=-1, dims=[2, 3])
                prediction = (prediction + routput) / 2

            if opts["device"] == "cpu":
                prediction = prediction.squeeze(1).detach().numpy()
            else:
                prediction = prediction.squeeze(1).detach().cpu().numpy()
            
            
            #Load and save temp prediction
            pred_path = task_path.joinpath(f"{filename_base}.npy")
            prediction = np.concatenate([np.load(pred_path), prediction], axis=0)

            np.save(str(pred_path), prediction)

        del model
    # shutil.rmtree(temp_path.absolute())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--submission-path", default="data/ensemble_preds")
    parser.add_argument("--data-type", default="validation", help="validation or test")
    parser.add_argument("--task", type=int, default=1, help="Which task you are submitting for")

    parser.add_argument("--config", type=str, default="config/main.yaml", help="Config")
    parser.add_argument("--device", type=str, default="cuda", help="Which device the inference should run on")

    parser.add_argument("--data-ratio", type=float, default=1.0, help="Percentage of the whole dataset that is used")

    args = parser.parse_args()


    pt_share_links1 = [
        (
            "https://drive.google.com/file/d/1cdFRQ12R5MziMVrE1vNKTqRL3_1Toh3x/view?usp=share_link",
            "https://drive.google.com/file/d/1jKOqHwRWNFYF67P9VFgJV9zDQ1-VJsTd/view?usp=share_link"
        ),
        (
            "https://drive.google.com/file/d/1173maCZwYTYcbbML0aGY0MCh4WXHe6p2/view?usp=sharing",
            "https://drive.google.com/file/d/1CteVOt7fatjHdobtuk3toaRMS6nYEU7s/view?usp=share_link"
        ),
        (
            "https://drive.google.com/file/d/1Aqr9LnAZHMKsOZkoUp583Q3VzuTYaaUV/view?usp=share_link",
            "https://drive.google.com/file/d/1-id3l8kd1QwBOE6KDLb4FBadrUKypFpT/view?usp=share_link"
        ),
        (
            "https://drive.google.com/file/d/16xFkFkTgaYK5a96P1larK25749nF3G_g/view?usp=share_link",
            "https://drive.google.com/file/d/133TgE-Ao731rpw0pjgWn_LVoxHXBhXmt/view?usp=share_link"
        ),
        (
            "https://drive.google.com/file/d/1DGL9xdN-E8ibjkpQAq2sfHo-mOwFJnFs/view?usp=share_link",
            "https://drive.google.com/file/d/1skM8EukONBtx2b7x1qm1XJI2AXORaEN5/view?usp=share_link"
        ),
        ( 
            "https://drive.google.com/file/d/1Hp6HAT7bTYMAt4MtQwsavNGIsUJ0z0m-/view?usp=share_link",
            "https://drive.google.com/file/d/1y1Fr1pOORX1A39ZzrDEf7VxgxlUCUPaV/view?usp=share_link"
        ),
        # (
        #     "https://drive.google.com/file/d/1C-b0TUORvbrCuDfCL6bkeTOgrw2P0eru/view?usp=share_link",
        #     "https://drive.google.com/file/d/1SwA28lxSz1MZTCLWpVFenUrFSH98kGx4/view?usp=share_link"
        # ),
        (
            "https://drive.google.com/file/d/155y9VfHUaJY5ed8Rzo4chDJ6Yx8fJ4GQ/view?usp=share_link",
            "https://drive.google.com/file/d/1NCmt2N6SToatfwSwNyZD0_H9jPhXHksM/view?usp=share_link"
        )
    ]
    
    pt_share_links2 = [
            (
            "https://drive.google.com/file/d/1iBmM3CuvKx-4CY1-7gU9jTWeuUXqvfRn/view?usp=share_link", # cp mapai resnest
            "https://drive.google.com/file/d/1yQctpXyuBgfR1gzc72yrdKRpEGD8Ojdo/view?usp=share_link" # opts
            ),
            (
            "https://drive.google.com/file/d/1EbSTbVADnwwuR6AYXXjK0nTtXjPeLAyU/view?usp=share_link", # mapai eb1
            "https://drive.google.com/file/d/1FQdxYBGYkr1_NTxtFwS0XrEs2dLybiza/view?usp=share_link"
            ),
            (
            "https://drive.google.com/file/d/1NtUP5QwhBglf0zSlQ0rRvIr2o4qMH8Bk/view?usp=share_link", # resnest reclassified
            "https://drive.google.com/file/d/1ZX0H4WTdz0caX0lLNJ66D6mcoC5HsFO-/view?usp=share_link"
            ),
            (
                "https://drive.google.com/file/d/1W8PF0sKh2PVU-SzXeLJvXvheWaVF6w_a/view?usp=share_link", # eb1 lidar masks
                "https://drive.google.com/file/d/1whc8TQzFCaFHqBHer1vXWf1tjhPzCtZK/view?usp=share_link"
            ),
            (
                "https://drive.google.com/file/d/1RItf98I8fFOjuepgp1mPNbnXdwidbAo-/view?usp=share_link", # resnest lidar masks
                "https://drive.google.com/file/d/1jn_0qNke165sQCoue6PUwxkHm_bJ_32Z/view?usp=share_link"
            ),
            (
                "https://drive.google.com/file/d/1Wk83hV1rH9sLIJG8ggiMvhDfCPjRDyJb/view?usp=share_link", # eb2 reclassified
                "https://drive.google.com/file/d/1b0OumKfiXSMhUyqefAvbIyqLPKdnpOr-/view?usp=share_link"
            ),
            (
                "https://drive.google.com/file/d/1enih5gy9X46g_JMkfT8jYR5ssg_XNgel/view?usp=share_link",
                "https://drive.google.com/file/d/1p-XDUIbu-MbuISBqM43nV8JNqV_97UC1/view?usp=share_link"
            ),
            (
                "https://drive.google.com/file/d/1JUbNYl3Z0zFMYmTs6aILj-qRKOSdT5kh/view?usp=share_link",
                "https://drive.google.com/file/d/1-QthazPuHqbJdTSen8I80r3Cug_xweKA/view?usp=share_link"
            ),
            (
                "https://drive.google.com/file/d/1WYcXT6j7o7fL4roSZeTJ3K7NY9K5xwZd/view?usp=share_link",
                "https://drive.google.com/file/d/1-kON-6jE9Yi2uADvs7ep6GFaQZpGIOv5/view?usp=share_link"
            ),
        ]
    
    if args.task == 1:
        main(args, pt_share_links1)
    elif args.task == 2:
        main(args, pt_share_links2)
    else:
        args.task = 1
        main(args, pt_share_links1)

        args.task = 2
        main(args, pt_share_links2)
