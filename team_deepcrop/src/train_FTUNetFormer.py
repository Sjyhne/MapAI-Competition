import os

import numpy as np
from yaml import load, dump, Loader, Dumper
from tqdm import tqdm
import torch
import torchvision
from tabulate import tabulate
import wandb
from torch import nn
import argparse
import time

from competition_toolkit.dataloader import create_dataloader
from utils import create_run_dir, store_model_weights, record_scores

from competition_toolkit.eval_functions import calculate_score
from models import unet_resnet18

from models.geoseg.losses import *
# from models.geoseg.datasets.uavid_dataset import *
# from models.geoseg.models.UNetFormer import UNetFormer
from models.geoseg.models.FTUNetFormer import FTUNetFormer

from torch.utils.data import DataLoader
from models.geoseg.losses import *
# from models.geoseg.datasets.vaihingen_dataset import *
from models.geoseg.models.FTUNetFormer import ft_unetformer
from catalyst.contrib.nn import Lookahead
## new version
# from catalyst.contrib.optimizers.lookahead import Lookahead

from catalyst import utils
import ipdb

def cal_metrics(output, label, use_aux_loss):
    # use_aux_loss = True
    # ipdb.set_trace()

    if use_aux_loss:
        output = nn.Softmax(dim=1)(output[0])
    else:
        output = nn.Softmax(dim=1)(output)
    output = output.argmax(dim=1)
    if device != "cpu":
        # print("using cpu")
        metrics = calculate_score(output.detach().cpu().numpy().astype(np.uint8),
                                       label.detach().cpu().numpy().astype(np.uint8))
    else:
        print("using cpu")
        metrics = calculate_score(output.detach().numpy().astype(np.uint8),
                                       label.detach().numpy().astype(np.uint8))
    return metrics

def test(opts, dataloader, model, lossfn):
    model.eval()

    device = opts["device"]

    losstotal = np.zeros((len(dataloader)), dtype=float)
    ioutotal = np.zeros((len(dataloader)), dtype=float)
    bioutotal = np.zeros((len(dataloader)), dtype=float)
    scoretotal = np.zeros((len(dataloader)), dtype=float)

    for idx, batch in tqdm(enumerate(dataloader), leave=False, total=len(dataloader), desc="Test"):
        image, label, filename = batch
        image = image.to(device)
        label = label.to(device)

        # output = model(image)["out"]
        output = model(image)
        # ipdb.set_trace()

        loss = lossfn(output, label).item()
        use_aux_loss = False
        metrics = cal_metrics(output, label, use_aux_loss)

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
    print(device)

    # The current model should be swapped with a different one of your choice
    # model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=opts["num_classes"])
    # model = torchvision.models.segmentation.fcn_resnet101(pretrained=False, num_classes=opts["num_classes"], pretrained_backbone=True)

    # model = torchvision.models.segmentationß.deeplabv3_resnet50(pretrained=False, num_classes=opts["num_classes"], pretrained_backbone=True)

    # model = unet_resnet18.ResNetUNet(n_class=2)
    num_classes = 2
    ignore_index = None
    model = FTUNetFormer(num_classes=2)

    if opts["task"] == 2:
        print("process for the task 2")
        new_conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.backbone.conv1 = new_conv1

    model.to(device)
    model = model.float()

#### default parameter for lr with pre-trained model
    # lr = 6e-4
    # weight_decay = 2.5e-4
    # backbone_lr = 6e-5
    # backbone_weight_decay = 2.5e-4
    # accumulate_n = 1

    ## change to the train from scratch
    lr = opts["lr"]
    weight_decay = 2.5e-3
    backbone_lr = lr
    backbone_weight_decay = 2.5e-3
    accumulate_n = 1
    net = ft_unetformer(pretrained=True,num_classes=num_classes, decoder_channels=256)

    layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
    net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
    base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
    optimizer = Lookahead(base_optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    # optimizer = torch.optim.Adam(model.parameters(), lr=opts["lr"])
    # lossfn = torch.nn.CrossEntropyLoss()
    # lossfn = UnetFormerLoss()
    lossfn = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                     DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)


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

            # import ipdb; ipdb.set_trace()
            # output = model(image)["out"]
            output = model(image)


            loss = lossfn(output, label)
            wandb.log({"loss":loss})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### backward from FTUnetFormer


            # return {"loss": loss}

            lossitem = loss.item()
            # output = torch.argmax(torch.softmax(output, dim=1), dim=1)
            use_aux_loss = False

            trainmetrics = cal_metrics(output, label, use_aux_loss )

            losstotal[idx] = lossitem
            ioutotal[idx] = trainmetrics["iou"]
            bioutotal[idx] = trainmetrics["biou"]
            scoretotal[idx] = trainmetrics["score"]

        testloss, testiou, testbiou, testscore = test(opts, valloader, model, lossfn)
        trainloss = round(losstotal.mean(), 4)
        trainiou = round(ioutotal.mean(), 4)
        trainbiou = round(bioutotal.mean(), 4)
        trainscore = round(scoretotal.mean(), 4)

        wandb.log({"loss":loss, "testloss": testloss, "trainloss": trainloss, "testiou": testiou, "testbiou": testbiou,"trainbiou":trainbiou, "trainiou": trainiou, "trainscore": trainscore})
        wandb.watch(model)

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

    parser.add_argument("--epochs", type=int, default=80, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate used during training")
    parser.add_argument("--config", type=str, default="config/data.yaml", help="Configuration file to be used")
    # parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--data_ratio", type=float, default=1.0,
                        help="Percentage of the whole dataset that is used")
    parser.add_argument("--name", type=str, default="run_seg")
    parser.add_argument("--wandb", type=str, default="disabled")

    args = parser.parse_args()
    wandb.init(
        mode=args.wandb,
                project="BuildingSeg",
               entity="lennylei",
               config={
                   "epochs": args.epochs,
                   "lr":args.lr
               }
               )


    wandb.run.name = args.name

    # Import config
    opts = load(open(args.config, "r"), Loader)

    # Combine args and opts in single dict
    try:
        opts = opts | vars(args)
    except Exception as e:
        opts = {**opts, **vars(args)}
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    opts["device"] = device
    print("Opts:", opts)

    rundir = create_run_dir(opts)
    opts["rundir"] = rundir
    dump(opts, open(os.path.join(rundir, "opts.yaml"), "w"), Dumper)

    train(opts)
    wandb.finish()
