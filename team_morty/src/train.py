import os

import numpy as np
from yaml import load, dump, Loader, Dumper
from tqdm import tqdm
import torch
from tabulate import tabulate
import argparse
import time
from augmentation.dataloader import create_dataloader
from ai_models.create_models import load_unet, load_resnet101, load_resnet50
from utils import create_run_dir, store_model_weights, record_scores
from competition_toolkit.eval_functions import calculate_score
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses.dice import DiceLoss

def test(opts, dataloader, model, lossfn, get_output):
    model.eval()

    device = opts["device"]

    losstotal = np.zeros((len(dataloader)), dtype=float)
    ioutotal = np.zeros((len(dataloader)), dtype=float)
    bioutotal = np.zeros((len(dataloader)), dtype=float)
    scoretotal = np.zeros((len(dataloader)), dtype=float)

    for idx, batch in tqdm(enumerate(dataloader), leave=False, total=len(dataloader), desc="Test"):
        # if idx > 30:
        #     break
        image, label, filename = batch
        image = image.to(device)
        label = label.to(device)

        output = model(image)[get_output]

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
    device = opts["device"]

    model, get_output = load_unet(opts)
    # model, get_output = load_resnet50(opts)
    # model, get_output = load_resnet50(opts, pretrained=True)
    # model, get_output = load_resnet101(opts)


    if opts["task"] == 2:
        new_conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.backbone.conv1 = new_conv1

    model.to(device)
    model = model.float()

    optimizer = torch.optim.Adam(model.parameters(), lr=opts["lr"])
    # lossfn = torch.nn.CrossEntropyLoss()
    # lossfn = torch.nn.BCELoss()
    lossfn = DiceLoss(mode="multiclass")


    epochs = opts["epochs"]

    trainloader = create_dataloader(opts, "train")
    valloader = create_dataloader(opts, "validation")

    bestscore = 0

    for e in range(epochs):

        model.train()

        losstotal = np.zeros((len(trainloader)), dtype=float)
        scoretotal = np.zeros((len(trainloader)), dtype=float)
        ioutotal = np.zeros((len(trainloader)), dtype=float)
        bioutotal = np.zeros((len(trainloader)), dtype=float)

        stime = time.time()


        for idx, batch in tqdm(enumerate(trainloader), leave=True, total=len(trainloader), desc="Train", position=0):
            image, label, filename = batch
            image = image.to(device)
            label = label.to(device)
            output = model(image)[get_output]

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

        testloss, testiou, testbiou, testscore = test(opts, valloader, model, lossfn, get_output)
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
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate used during training")
    parser.add_argument("--config", type=str, default="team_morty/src/config/data.yaml", help="Configuration file to be used")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--data_ratio", type=float, default=1.0,
                        help="Percentage of the whole dataset that is used")

    args = parser.parse_args()

    # Import config
    opts = load(open(args.config, "r"), Loader)

    # Combine args and opts in single dict
    try:
        opts = opts | vars(args)
    except Exception as e:
        opts = {**opts, **vars(args)}

    print("Opts:", opts)

    rundir = create_run_dir(opts)
    opts["rundir"] = rundir
    dump(opts, open(os.path.join(rundir, "opts.yaml"), "w"), Dumper)

    train(opts)
