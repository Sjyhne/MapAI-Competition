import os

import numpy as np
from yaml import load, dump, Loader, Dumper
from tqdm import tqdm
import torch
import torchvision
from tabulate import tabulate
from kornia.morphology import erosion, dilation
import argparse
import time
# import cv2
#from competition_toolkit.dataloader import create_dataloader
from custom_dataloader import create_dataloader
from utils import create_run_dir, store_model_weights, record_scores, get_model, get_optimizer, get_losses, get_scheduler, get_aug_names
from transforms import valid_transform, get_lidar_transform
from competition_toolkit.eval_functions import calculate_score
from multiclass_metrics import calculate_multiclass_score
import transforms

transforms = transforms.__dict__
def test(dataloader, model, lossfn, device, aux_loss=None, interpolation_mode=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True, aux_head=False, erode=False):
    model.eval()

    device = device


    losstotal = np.zeros((len(dataloader)), dtype=float)
    lossesseg = np.zeros((len(dataloader)), dtype=float)
    lossesaux = np.zeros((len(dataloader)), dtype=float)
    ioutotal = np.zeros((len(dataloader)), dtype=float)
    bioutotal = np.zeros((len(dataloader)), dtype=float)
    scoretotal = np.zeros((len(dataloader)), dtype=float)
    filenames = []

    for idx, batch in tqdm(enumerate(dataloader), leave=False, total=len(dataloader), desc="Test"):
        if aux_head:
            filename, image, label, aux_label = batch.values() 
            aux_label = aux_label.to(device)
        else:
            filename, image, label = batch.values()
        image = image.to(device)
        label = label.long().to(device)
        
        loss_aux = 0
        output = model(image)
        if aux_head:
            output, aux_pred = output
            loss_aux = aux_loss(aux_pred, aux_label).item()
            lossesaux[idx] = loss_aux

        if label.shape[-2:] != output.shape[-2:]:
            output = torchvision.transforms.functional.resize(output, (500, 500), interpolation=interpolation_mode, antialias=antialias)
        losseg = lossfn(output, label).item()
        lossesseg[idx] = losseg
        losstotal[idx] = loss_aux + losseg

        num_classes = output.shape[1]
        if output.shape[1] > 1:
            output = torch.argmax(torch.softmax(output, dim=1), dim=1)
            if num_classes == 3:
                output[output == 2.0] = 0.0
            elif num_classes == 4:
                output[output == 2.0] = 1.0
                output[output == 3.0] = 0.0
        else:
            output = torch.round(torch.sigmoid(output)).squeeze(1)
        label = label.squeeze(1)

        if erode:
            kernel = torch.ones(5, 5).to(device)
            output = erosion(output.unsqueeze(1), kernel)
            output = dilation(output, kernel).squeeze(1)

        if device != "cpu":
            metrics = calculate_score(output.detach().cpu().numpy().astype(np.uint8),
                                    label.detach().cpu().numpy().astype(np.uint8))
        else:
            metrics = calculate_score(output.detach().numpy().astype(np.uint8), label.detach().numpy().astype(np.uint8))

        ioutotal[idx] = metrics["iou"]
        bioutotal[idx] = metrics["biou"]
        scoretotal[idx] = metrics["score"]
        # if metrics["score"] < 1.0:
        #     filenames.append((filename, metrics["score"]))
        #     img = output.detach().cpu().numpy().astype(np.uint8)
        #     for img_idx in range(img.shape[0]):
        #         cv2.imwrite("datatest/" + filename[img_idx], img[img_idx] * 255)

    # filenames = sorted(filenames, key=lambda x: x[1])
    # for files, scores in filenames[:10]:
    #     print(files, scores)
    # for files, scores in filenames[-10:]:
    #     print(files, scores)

    iou = round(ioutotal.mean(), 4)
    loss = round(losstotal.mean(), 4)
    lossesseg = round(lossesseg.mean(), 4)
    lossesaux = round(lossesaux.mean(), 4)
    biou = round(bioutotal.mean(), 4)
    score = round(scoretotal.mean(), 4)
    return loss, lossesaux, lossesseg, iou, biou, score


def train(opts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    # The current model should be swapped with a different one of your choice

    model = get_model(opts)
    if opts["weights"] is not None:
        model.load_state_dict(torch.load(opts["weights"]))

    model.to(device)
    model = model.float()
    aux_head = opts["model"]["aux_head"]

    optimizer = get_optimizer(opts, model)
    scheduler = get_scheduler(opts, optimizer)
    lossfn = get_losses(opts)
    aux_loss = torch.nn.BCEWithLogitsLoss()

    epochs = opts["train"]["epochs"]


    augmentation_cfg = opts["augmentation"]
    initial_transform = transforms[augmentation_cfg.get("initial", "normal")](opts["imagesize"])
    aug_list = get_aug_names(opts, augmentation_cfg, transforms)

    lidar_transform = None
    if opts["task"] == 2:
        lidar_transform = get_lidar_transform(opts)

    trainloader = create_dataloader(opts, "train", transforms=(initial_transform, lidar_transform), aux_head_labels=aux_head)
    valloader = create_dataloader(opts, "validation", transforms=(valid_transform, lidar_transform), aux_head_labels=aux_head)

    bestscore = 0

    print("Training with augmentation schedule: ", aug_list)

    print("Initial learning rate", scheduler.get_lr())
    for e in range(epochs):

        model.train()

        losstotal = np.zeros((len(trainloader)), dtype=float)
        losses_aux = np.zeros((len(trainloader)), dtype=float)
        losses_seg = np.zeros((len(trainloader)), dtype=float)
        scoretotal = np.zeros((len(trainloader)), dtype=float)
        ioutotal = np.zeros((len(trainloader)), dtype=float)
        bioutotal = np.zeros((len(trainloader)), dtype=float)

        stime = time.time()

        if e >= augmentation_cfg["warmup_epochs"]:
            aug = aug_list[e]
            print(f"Using the {aug} transform")
            new_transform = transforms[aug](opts["imagesize"])
            trainloader.dataset.set_transform(new_transform)

        pbar = tqdm(enumerate(trainloader), leave=True, total=len(trainloader), desc="Train", position=0)
        for idx, batch in pbar:
            if aux_head:
                filename, image, label, aux_label = batch.values() 
                aux_label = aux_label.to(device)
            else:
                filename, image, label = batch.values()
            image = image.to(device)
            label = label.long().to(device) 


            loss_aux = 0
            output = model(image)
            
            if opts["task"] == 3:
                pass
                # output = output.squeeze(dim=1)
                # img = output.squeeze().detach().cpu().numpy().copy()
                # img = np.exp(img) + math.e
                # maximum = np.max(img)
                # img = img * (255.0/maximum)

                # img = np.stack([img, img, img], axis=-1).astype(np.uint8)
                # print(img.shape)
                # cv2.imwrite(opts["rundir"] + f"/pred{idx}.png", img)

                # pred_img = image.squeeze().cpu().numpy() * 255.0
                # cv2.imwrite(opts["rundir"] + f"/pred{idx}img.png", pred_img.transpose(1, 2, 0).astype(np.uint8))
                # if idx == 20:
                #     exit()
            elif aux_head:
                output, aux_pred = output
                loss_aux = aux_loss(aux_pred, aux_label)
                losses_aux[idx] = loss_aux.item()

            loss_seg = lossfn(output, label)
            losses_seg[idx] = loss_seg.item()

            loss = loss_aux + loss_seg

            pbar.set_description("Train - Loss {:.4f}".format(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossitem = loss.item()
            losstotal[idx] = lossitem
            if output.shape[1] > 1:
                output = torch.argmax(torch.softmax(output, dim=1), dim=1)
            else:
                output = torch.round(torch.sigmoid(output)).squeeze(1)
            label = label.squeeze(1)

            if device != "cpu":
                trainmetrics = calculate_multiclass_score(output.detach().cpu().numpy().astype(np.uint8),
                                               label.detach().cpu().numpy().astype(np.uint8), opts["num_classes"])
            else:
                trainmetrics = calculate_multiclass_score(output.detach().numpy().astype(np.uint8),
                                               label.detach().numpy().astype(np.uint8), opts["num_classes"])

            ioutotal[idx] = trainmetrics["iou"]
            bioutotal[idx] = trainmetrics["biou"]
            scoretotal[idx] = trainmetrics["score"]
        scheduler.step()

        testloss, testlossaux, testlosseg, testiou, testbiou, testscore = test(valloader, model, lossfn, device, aux_loss=aux_loss, aux_head=aux_head, erode=opts["erode_val_preds"])
        trainloss = round(losstotal.mean(), 4)
        trainlosseg = round(losses_seg.mean(), 4)
        trainlossaux = round(losses_aux.mean(), 4)
        trainiou = round(ioutotal.mean(), 4)
        trainbiou = round(bioutotal.mean(), 4)
        trainscore = round(scoretotal.mean(), 4)

        if testscore > bestscore:
            bestscore = testscore
            print("new best score:", bestscore, "- saving model weights")
            store_model_weights(opts, model, f"best", epoch=e)
        else:
            store_model_weights(opts, model, f"last", epoch=e)
        print(tabulate(
            [["train", trainloss, trainlossaux, trainlosseg, trainiou, trainbiou, trainscore], ["test", testloss, testlossaux, testlosseg, testiou, testbiou, testscore]],
            headers=["Type", "LossTotal", "LossAux", "LossSeg", "IoU", "BIoU", "Score"]))

        print(f"Starting epoch {e + 1} with new learning rate", scheduler.get_lr())
        print("")
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
    parser.add_argument("--weights", type=str, default=None)

    args = parser.parse_args()

    singleclass = {}
    threeclass = {
        "use_lidar_in_mask": True,
        "num_classes": 3,
        "dataset": "lidar_masks"
    }
    fourclass = {
        "num_classes": 4,
        "data_dirs": {
            "masks_train": "masks_reclassified"
        },
        "dataset": "masks_reclassified"
    }
    multihead = {
        "model": {
            "aux_head": True
        },
        "dataset": "multihead"
    }

    ds_list = [threeclass, fourclass, multihead, singleclass]

    for idx, ds_params in enumerate(ds_list):
        # Import config
        opts = load(open(args.config, "r"), Loader)

        # Combine args and opts in single dict
        try:
            opts = opts | vars(args)
        except Exception as e:
            opts = {**opts, **vars(args)}

        if opts["use_lidar_in_mask"]:
            opts["num_classes"] = 3

        if idx < 2:
            opts["training"]["losses"]["DiceLoss"]["init_params"]["mode"] = "multiclass"

        for key, value in ds_params.items():
            if type(value) == dict:
                opts[key][list(value.keys())[0]] = list(value.values())[0]
            else:
                opts[key] = value
        
        rundir = create_run_dir(opts, opts.get("dataset", ""))
        opts["rundir"] = rundir
        print("Opts:", opts)
        dump(opts, open(os.path.join(rundir, "opts.yaml"), "w"), Dumper)

        train(opts)
