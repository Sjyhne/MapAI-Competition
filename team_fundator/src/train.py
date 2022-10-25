import os
from cv2 import transform

import numpy as np
from yaml import load, dump, Loader, Dumper
from tqdm import tqdm
import torch
import torchvision
from tabulate import tabulate

import argparse
import time
import segmentation_models_pytorch as smp
#from competition_toolkit.dataloader import create_dataloader
from custom_dataloader import create_dataloader
from utils import create_run_dir, store_model_weights, record_scores, get_model, get_optimizer, get_losses
from transforms import valid_transform, get_lidar_transform
from competition_toolkit.eval_functions import calculate_score
import transforms

transforms = transforms.__dict__
print(transforms.keys())
def test(dataloader, model, lossfn, device):
    model.eval()

    device = device

    losstotal = np.zeros((len(dataloader)), dtype=float)
    ioutotal = np.zeros((len(dataloader)), dtype=float)
    bioutotal = np.zeros((len(dataloader)), dtype=float)
    scoretotal = np.zeros((len(dataloader)), dtype=float)

    for idx, batch in tqdm(enumerate(dataloader), leave=False, total=len(dataloader), desc="Test"):
        image, label = batch.values()
        image = image.to(device)
        label = label.long().to(device)
        
        output = model(image)

        loss = lossfn(output, label).item()

        output = torch.argmax(torch.softmax(output, dim=1), dim=1)
        if device != "cpu":
            metrics = calculate_score(output.detach().cpu().numpy().astype(np.uint8),
                                      label.detach().cpu().numpy().astype(np.uint8))
        else:
            metrics = calculate_score(output.detach().numpy().astype(np.uint8), label.detach().numpy().astype(np.uint8))

        losstotal[idx] = loss
        ioutotal[idx] = metrics["iou"]
        bioutotal[idx] = metrics["biou"]
        scoretotal[idx] = metrics["score"]

    loss = round(losstotal.mean(), 4)
    iou = round(ioutotal.mean(), 4)
    biou = round(bioutotal.mean(), 4)
    score = round(scoretotal.mean(), 4)

    return loss, iou, biou, score


def train(opts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    # The current model should be swapped with a different one of your choice

    model = get_model(opts)

    model.to(device)
    model = model.float()

    optimizer = get_optimizer(opts, model)
    lossfn = get_losses(opts)

    epochs = opts["train"]["epochs"]

    train_transform = transforms[opts.get('augmentation', 'valid_transform')]
    print(train_transform)



    lidar_transform = None
    if opts["task"] == 2:
        lidar_transform = get_lidar_transform(opts)

    trainloader = create_dataloader(opts, "train", transforms=(train_transform, lidar_transform))
    valloader = create_dataloader(opts, "validation", transforms=(valid_transform, lidar_transform))

    bestscore = 0

    for e in range(epochs):

        model.train()

        losstotal = np.zeros((len(trainloader)), dtype=float)
        scoretotal = np.zeros((len(trainloader)), dtype=float)
        ioutotal = np.zeros((len(trainloader)), dtype=float)
        bioutotal = np.zeros((len(trainloader)), dtype=float)

        stime = time.time()

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
