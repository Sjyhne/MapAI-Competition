import os

import numpy as np
from yaml import load, dump, Loader, Dumper
from tqdm import tqdm
import torch
import torchvision
from tabulate import tabulate

import argparse
import time
#from competition_toolkit.dataloader import create_dataloader
from custom_dataloader import create_dataloader
from utils import create_run_dir, store_model_weights, record_scores, get_model, get_optimizer, get_losses, get_scheduler, get_aug_names
from transforms import valid_transform, get_lidar_transform
from competition_toolkit.eval_functions import calculate_score
import transforms

transforms = transforms.__dict__
print(transforms.keys())
def test(dataloader, model, lossfn, device, interpolation_mode=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True):
    model.eval()

    device = device


    losstotal = np.zeros((len(dataloader)), dtype=float)
    ioutotal = np.zeros((len(dataloader)), dtype=float)
    bioutotal = np.zeros((len(dataloader)), dtype=float)
    scoretotal = np.zeros((len(dataloader)), dtype=float)

    for idx, batch in tqdm(enumerate(dataloader), leave=False, total=len(dataloader), desc="Test"):
        image, label = batch.values()
        image = image.to(device)
        label = label.long().to(device) if opts["task"] != 3 else label.to(device)
        

        output = model(image)
        if opts["task"] == 3:
            output = output.squeeze(dim=1)
        if label.shape[-2:] != output.shape[-2:]:
            output = torchvision.transforms.functional.resize(output, (500, 500), interpolation=interpolation_mode, antialias=antialias)

        loss = lossfn(output, label).item()
        losstotal[idx] = loss

        if opts["task"] == 3:
            continue

        output = torch.argmax(torch.softmax(output, dim=1), dim=1)
        if device != "cpu":
            metrics = calculate_score(output.detach().cpu().numpy().astype(np.uint8),
                                    label.detach().cpu().numpy().astype(np.uint8))
        else:
            metrics = calculate_score(output.detach().numpy().astype(np.uint8), label.detach().numpy().astype(np.uint8))

        ioutotal[idx] = metrics["iou"]
        bioutotal[idx] = metrics["biou"]
        scoretotal[idx] = metrics["score"]


    iou = round(ioutotal[0].mean(), 4)
    loss = round(losstotal[0].mean(), 4)
    biou = round(bioutotal[0].mean(), 4)
    score = round(scoretotal[0].mean(), 4)
    return loss, iou, biou, score


def train(opts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    # The current model should be swapped with a different one of your choice

    model = get_model(opts)

    model.to(device)
    model = model.float()

    optimizer = get_optimizer(opts, model)
    scheduler = get_scheduler(opts, optimizer)
    lossfn = get_losses(opts)

    epochs = opts["train"]["epochs"]


    augmentation_cfg = opts["augmentation"]
    initial_transform = transforms[augmentation_cfg.get("initial", "normal")]
    aug_list = get_aug_names(opts, augmentation_cfg, transforms)



    lidar_transform = None
    if opts["task"] >= 2:
        lidar_transform = get_lidar_transform(opts)

    trainloader = create_dataloader(opts, "train", transforms=(initial_transform, lidar_transform))
    valloader = create_dataloader(opts, "validation", transforms=(valid_transform, lidar_transform))

    bestscore = 0


    print("Training with augmentation schedule: ", aug_list)

    print("Initial learning rate", scheduler.get_lr())
    for e in range(epochs):

        model.train()

        losstotal = np.zeros((len(trainloader)), dtype=float)
        scoretotal = np.zeros((len(trainloader)), dtype=float)
        ioutotal = np.zeros((len(trainloader)), dtype=float)
        bioutotal = np.zeros((len(trainloader)), dtype=float)

        stime = time.time()

        if e >= augmentation_cfg["warmup_epochs"]:
            aug = aug_list[e]
            print(f"Using the {aug} transform")
            new_transform = transforms[aug]
            trainloader.dataset.set_transform(new_transform)

        for idx, batch in tqdm(enumerate(trainloader), leave=True, total=len(trainloader), desc="Train", position=0):
            image, label = batch.values()
            image = image.to(device)
            label = label.long().to(device) if opts["task"] != 3 else label.to(device)


            output = model(image)
            if opts["task"] == 3:
                output = output.squeeze(dim=1)
            loss = lossfn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossitem = loss.item()
            losstotal[idx] = lossitem
            if opts["task"] == 3:
                continue

            output = torch.argmax(torch.softmax(output, dim=1), dim=1)
            if device != "cpu":
                trainmetrics = calculate_score(output.detach().cpu().numpy().astype(np.uint8),
                                               label.detach().cpu().numpy().astype(np.uint8))
            else:
                trainmetrics = calculate_score(output.detach().numpy().astype(np.uint8),
                                               label.detach().numpy().astype(np.uint8))

            ioutotal[idx] = trainmetrics["iou"]
            bioutotal[idx] = trainmetrics["biou"]
            scoretotal[idx] = trainmetrics["score"]
        scheduler.step()

        testloss, testiou, testbiou, testscore = test(valloader, model, lossfn, device)
        trainloss = round(losstotal.mean(), 4)
        trainiou = round(ioutotal.mean(), 4)
        trainbiou = round(bioutotal.mean(), 4)
        trainscore = round(scoretotal.mean(), 4)

        if testscore > bestscore:
            bestscore = testscore
            print("new best score:", bestscore, "- saving model weights")
            store_model_weights(opts, model, f"best", epoch=e)
        else:
            store_model_weights(opts, model, f"last", epoch=e)
        print("New learning rate", scheduler.get_lr())
        print("")
        print(tabulate(
            [["train", trainloss, trainiou, trainbiou, trainscore], ["test", testloss, testiou, testbiou, testscore]],
            headers=["Type", "Loss", "IoU", "BIoU", "Score"]))

        scoredict = {
            "epoch": e,
            "trainloss": trainloss,
            "testloss": testloss,
            "trainiou": trainiou,
            "testiou": testiou,
            "trainbiou": trainbiou,
            "testbiou": testbiou,
            "trainscore": trainscore,
            "testscore": testscore
        }

        record_scores(opts, scoredict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Training a segmentation model")

    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--config", type=str, default="config/data.yaml", help="Configuration file to be used")
    parser.add_argument("--task", type=int, default=1)

    args = parser.parse_args()

    # Import config
    opts = load(open(args.config, "r"), Loader)

    # Combine args and opts in single dict
    try:
        opts = opts | vars(args)
    except Exception as e:
        opts = {**opts, **vars(args)}

    print("Opts:", opts)

    rundir = create_run_dir(opts, opts.get("dataset", ""))
    opts["rundir"] = rundir
    dump(opts, open(os.path.join(rundir, "opts.yaml"), "w"), Dumper)

    train(opts)
