import argparse
import torch

from tqdm import tqdm
from custom_dataloader import create_dataloader
from ensemble_model import EnsembleModel, load_models_from_runs
from transforms import valid_transform, LidarAugComposer
import yaml
import pickle
from utils import get_dataset_config
import os

def test_ensemble(opts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models and create ensemble
    models, runs = load_models_from_runs(
        opts["run_folder"], "*"
    )
    savetask = opts['task'] if opts["task"] != 3 else 2

    model = EnsembleModel(models, sum_outputs=False)
    model.to(device)
    model.eval()

    # Load data
    lidar_transform = None
    if opts["task"] != 1:
        aug_getter = LidarAugComposer(opts)
        _, lidar_transform = aug_getter.get_transforms()

    opts["train"]["batchsize"] = 1
    opts["validation"]["batchsize"] = 1
    opts["use_lidar_in_mask"] = False

    for split in ["train", "validation"]:
        save_folder = f"./data/ensembles/task{savetask}/{opts['name']}/{split}"
        os.makedirs(save_folder)

        dataloader = create_dataloader(
            opts,
            split,
            transforms=(valid_transform, lidar_transform),
            aux_head_labels=False,
        )

        for idx, batch in tqdm(
            enumerate(dataloader), leave=False, total=len(dataloader), desc="Test"
        ):
            filename, image, _ = batch.values()
            image = image.to(device)

            output = model(image)
            model_preds = output["model_preds"]
            model_preds = [pred.squeeze(0).detach().numpy() for pred in model_preds]
            with open(save_folder + '/' + filename[0][:-3] + "pickle", 'wb') as handle:
                pickle.dump(model_preds, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Save ensemble model predictions")

    parser.add_argument(
        "--config",
        type=str,
        default="config/save_ensemble_preds.yaml",
        help="Configuration file to be used",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the ensemble",
    )
    parser.add_argument(
        "--run_folder",
        type=str,
        help="Folder to load the ensemble from",
    )
    parser.add_argument("--task", type=int, required=True)

    args = parser.parse_args()

    # Import config
    opts = yaml.load(open(args.config, "r"), yaml.Loader)
    
    data_opts = get_dataset_config(opts)

    opts.update(data_opts)
    # Combine args and opts in single dict
    try:
        opts = opts | vars(args)
    except Exception as e:
        opts = {**opts, **vars(args)}

    print("Opts:", opts)

    test_ensemble(opts)
