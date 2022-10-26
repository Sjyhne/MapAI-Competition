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
from utils import create_run_dir, store_model_weights, record_scores, get_model, get_optimizer, get_losses, get_scheduler
from transforms import valid_transform, get_lidar_transform
from competition_toolkit.eval_functions import calculate_score
import transforms

transforms = transforms.__dict__
print(transforms.keys())
def test(dataloader, model, lossfn, device, interpolation_mode=torchvision.transforms.InterpolationMode.BILINEAR):
    model.eval()

    device = device


    interpolation_modes = [torchvision.transforms.InterpolationMode.BICUBIC, torchvision.transforms.InterpolationMode.BILINEAR, torchvision.transforms.InterpolationMode.NEAREST]
    losstotal = [np.zeros((len(dataloader)), dtype=float) for i in range(5)]
    ioutotal = [np.zeros((len(dataloader)), dtype=float) for i in range(5)]
    bioutotal = [np.zeros((len(dataloader)), dtype=float) for i in range(5)]
    scoretotal = [np.zeros((len(dataloader)), dtype=float) for i in range(5)]

    for idx, batch in tqdm(enumerate(dataloader), leave=False, total=len(dataloader), desc="Test"):
        image, label = batch.values()
        image = image.to(device)
        label = label.long().to(device)
        
        big_output = model(image)
        im_idx = 0
        for interpolation_mode in interpolation_modes:
            for alias in [False, True]:
                output = torchvision.transforms.functional.resize(big_output, (500, 500), interpolation=interpolation_mode, antialias=alias)

                loss = lossfn(output, label).item()

                output = torch.argmax(torch.softmax(output, dim=1), dim=1)
                if device != "cpu":
                    metrics = calculate_score(output.detach().cpu().numpy().astype(np.uint8),
                                            label.detach().cpu().numpy().astype(np.uint8))
                else:
                    metrics = calculate_score(output.detach().numpy().astype(np.uint8), label.detach().numpy().astype(np.uint8))

                losstotal[im_idx][idx] = loss
                ioutotal[im_idx][idx] = metrics["iou"]
                bioutotal[im_idx][idx] = metrics["biou"]
                scoretotal[im_idx][idx] = metrics["score"]
                im_idx += 1
                if im_idx == 5:
                    break
            if im_idx == 5:
                break

    print(tabulate(
    [
        ["BICUBIC", losstotal[0].mean(), ioutotal[0].mean(), bioutotal[0].mean(), scoretotal[0].mean()],
        ["BICUBIC_ALIAS", losstotal[1].mean(), ioutotal[1].mean(), bioutotal[1].mean(), scoretotal[1].mean()],
        ["BILINEAR", losstotal[2].mean(), ioutotal[2].mean(), bioutotal[2].mean(), scoretotal[2].mean()],
        ["BILINEAR_Alias", losstotal[3].mean(), ioutotal[3].mean(), bioutotal[3].mean(), scoretotal[3].mean()],
        ["NEAREST", losstotal[4].mean(), ioutotal[4].mean(), bioutotal[4].mean(), scoretotal[4].mean()]],
    headers=["Type", "Loss", "IoU", "BIoU", "Score"]))

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
    train_transform = transforms[augmentation_cfg["initial"]]
    print(train_transform)



    lidar_transform = None
    if opts["task"] == 2:
        lidar_transform = get_lidar_transform(opts)

    trainloader = create_dataloader(opts, "train", transforms=(train_transform, lidar_transform))
    valloader = create_dataloader(opts, "validation", transforms=(valid_transform, lidar_transform))

    bestscore = 0
    
    print("Initial learning rate", scheduler.get_lr())
    for e in range(epochs):

        model.train()

        losstotal = np.zeros((len(trainloader)), dtype=float)
        scoretotal = np.zeros((len(trainloader)), dtype=float)
        ioutotal = np.zeros((len(trainloader)), dtype=float)
        bioutotal = np.zeros((len(trainloader)), dtype=float)

        stime = time.time()

        if e >= augmentation_cfg["warmup_epochs"]:
            augmentation_cycle = augmentation_cfg["cycle"]
            aug = augmentation_cycle[e % len(augmentation_cycle)]

            print(f"Using {aug} transforms")
            new_transform = transforms[aug]
            trainloader.dataset.set_transform(new_transform)

        for idx, batch in tqdm(enumerate(trainloader), leave=True, total=len(trainloader), desc="Train", position=0):
            image, label = batch.values()
            image = image.to(device)
            label = label.long().to(device)

            output = model(image)
            loss = lossfn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossitem = loss.item()
            output = torch.argmax(torch.softmax(output, dim=1), dim=1)
            if device != "cpu":
                trainmetrics = calculate_score(output.detach().cpu().numpy().astype(np.uint8),
                                               label.detach().cpu().numpy().astype(np.uint8))
            else:
                trainmetrics = calculate_score(output.detach().numpy().astype(np.uint8),
                                               label.detach().numpy().astype(np.uint8))

            losstotal[idx] = lossitem
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
