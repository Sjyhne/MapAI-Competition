import argparse
import numpy as np
from tabulate import tabulate

from tqdm import tqdm
from custom_dataloader import create_dataloader
from ensemble_model import EnsembleModel, load_models_from_runs
from transforms import valid_transform, LidarAugComposer
from competition_toolkit.eval_functions import calculate_score
import yaml
from utils import get_dataset_config


def test_ensemble(opts):
    device = opts["device"]

    # Load models and create ensemble
    models, runs = load_models_from_runs(
        opts.get("run_folder", "runs/task_1/mapai"), "*"
    )

    model = EnsembleModel(models)
    model.to(device)
    model.eval()

    # Load data
    lidar_transform = None
    if opts["task"] != 1:
        aug_getter = LidarAugComposer(opts)
        _, lidar_transform = aug_getter.get_transforms()

    dataloader = create_dataloader(
        opts,
        "validation",
        transforms=(valid_transform, lidar_transform),
        aux_head_labels=False,
    )

    ioutotal = np.zeros((len(models) + 1, len(dataloader)), dtype=float)
    bioutotal = np.zeros((len(models) + 1, len(dataloader)), dtype=float)
    scoretotal = np.zeros((len(models) + 1, len(dataloader)), dtype=float)

    for idx, batch in tqdm(
        enumerate(dataloader), leave=False, total=len(dataloader), desc="Test"
    ):
        filename, image, label = batch.values()
        image = image.to(device)
        label = label.long().to(device)

        output = model(image)
        model_preds = output["model_preds"]
        ensemble_preds = output["result"]
        model_preds = model_preds + [ensemble_preds]

        label = label.squeeze(1)

        for i in range(len(model_preds)):
            if device != "cpu":
                metrics = calculate_score(
                    model_preds[i].detach().cpu().numpy().astype(np.uint8),
                    label.detach().cpu().numpy().astype(np.uint8),
                )
            else:
                metrics = calculate_score(
                    model_preds[i].detach().numpy().astype(np.uint8),
                    label.detach().numpy().astype(np.uint8),
                )
            ioutotal[i, idx] = metrics["iou"]
            bioutotal[i, idx] = metrics["biou"]
            scoretotal[i, idx] = metrics["score"]

    iou = ioutotal.mean(axis=0)
    biou = bioutotal.mean(axis=0)
    score = scoretotal.mean(axis=0)
    mnames = runs + ["ensemble"]

    print("Achieved metrics:")
    print(
        tabulate(
            [
                [
                    mnames[i],
                    np.round(iou[i], 4),
                    np.round(biou[i], 4),
                    np.round(score[i], 4),
                ]
                for i in range(len(mnames))
            ],
            headers=["Run", "IoU", "BIoU", "Score"],
        )
    )
    print("Dumping these metrics to ensemble_metrics.csv...")
    with open("ensemble_metrics.csv", "w") as f:
        for i in range(len(mnames)):
            f.write(
                f"{mnames[i]},{np.round(iou[i], 4)},{np.round(biou[i], 4)},{np.round(score[i], 4)}\n"
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Testing of ensemble model")

    parser.add_argument(
        "--config",
        type=str,
        default="config/data.yaml",
        help="Configuration file to be used",
    )
    parser.add_argument("--task", type=int, default=1)

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
