import argparse
import numpy as np
from tabulate import tabulate
import torch

from tqdm import tqdm
from custom_dataloader import create_dataloader
from ensemble_model import EnsembleModel, load_models_from_runs
from transforms import valid_transform, get_lidar_transform
from competition_toolkit.eval_functions import calculate_score
import torchvision
import yaml


def test_ensemble(opts):
    device = opts["device"]
    antialias = opts.get("antialias", True)
    interpolation_mode = opts.get(
        "interpolation_mode", torchvision.transforms.InterpolationMode.BILINEAR
    )

    # Load models and create ensemble
    models, runs = load_models_from_runs(
        opts.get("run_folder", "runs/task_1/mapai"), "*"
    )

    model = EnsembleModel(models)
    model.to(device)
    model.eval()

    # Load data
    aux_head = opts["model"]["aux_head"]
    lidar_transform = None
    if opts["task"] == 2:
        lidar_transform = get_lidar_transform(opts)

    dataloader = create_dataloader(
        opts,
        "validation",
        transforms=(valid_transform, lidar_transform),
        aux_head_labels=aux_head,
    )

    ioutotal = np.zeros((len(models) + 1, len(dataloader)), dtype=float)
    bioutotal = np.zeros((len(models) + 1, len(dataloader)), dtype=float)
    scoretotal = np.zeros((len(models) + 1, len(dataloader)), dtype=float)

    for idx, batch in tqdm(
        enumerate(dataloader), leave=False, total=len(dataloader), desc="Test"
    ):
        if aux_head:
            image, label, aux_label = batch.values()
            aux_label = aux_label.to(device)
        else:
            image, label = batch.values()
        image = image.to(device)
        label = label.long().to(device)

        output = model(image)
        model_preds = output["model_preds"]
        ensemble_preds = output["result"]
        model_preds = model_preds + [ensemble_preds]

        if label.shape[-2:] != output.shape[-2:]:
            model_preds = [
                torchvision.transforms.functional.resize(
                    mp,
                    (500, 500),
                    interpolation=interpolation_mode,
                    antialias=antialias,
                )
                for mp in model_preds
            ]

        for i in range(len(model_preds)):
            if model_preds[i].shape[1] > 1:
                model_preds[i] = torch.argmax(
                    torch.softmax(model_preds[i], dim=1), dim=1
                )
            else:
                model_preds[i] = torch.round(torch.sigmoid(model_preds[i])).squeeze(1)
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
        "--epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/data.yaml",
        help="Configuration file to be used",
    )
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--weights", type=str, default=None)

    args = parser.parse_args()

    # Import config
    opts = yaml.load(open(args.config, "r"), yaml.Loader)

    # Combine args and opts in single dict
    try:
        opts = opts | vars(args)
    except Exception as e:
        opts = {**opts, **vars(args)}

    print("Opts:", opts)
    if opts["use_lidar_as_mask"]:
        opts["num_classes"] = 3

    test_ensemble(opts)
