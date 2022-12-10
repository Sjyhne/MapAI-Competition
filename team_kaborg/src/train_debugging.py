import os
from symbol import break_stmt

import numpy as np
from yaml import load, dump, Loader, Dumper
from tqdm import tqdm
import torch
from tabulate import tabulate
import argparse
import time
from augmentation.dataloader import create_dataloader
from ai_models.create_models import load_unet, load_resnet101, load_resnet50, load_deepvision_resnet101
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

    # model, get_output = load_unet(opts)
    # model, get_output = load_resnet50(opts)
    # model, get_output = load_resnet50(opts, pretrained=True)
    # model, get_output = load_resnet101(opts)
    model, get_output = load_deepvision_resnet101(opts, pretrained=True)

    #Load state dict from options resume
    if opts["resume"] is not None:
        model.load_state_dict(torch.load(opts["resume"], map_location=device))


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

    #trainloader = create_dataloader(opts, "train")
    valloader = create_dataloader(opts, "validation")
    #trainloader = create_dataloader(opts, "train")
        
    bestscore = 0
    if LOG_WANDB:
        wandb.watch(model)
    
    model.train()

    # losstotal = np.zeros((len(trainloader)), dtype=float)
    # scoretotal = np.zeros((len(trainloader)), dtype=float)
    # ioutotal = np.zeros((len(trainloader)), dtype=float)
    # bioutotal = np.zeros((len(trainloader)), dtype=float)


    testloss, testiou, testbiou, testscore = test(opts, valloader, model, lossfn, get_output)
    # trainloss = round(losstotal.mean(), 4)
    # trainiou = round(ioutotal.mean(), 4)
    # trainbiou = round(bioutotal.mean(), 4)
    # trainscore = round(scoretotal.mean(), 4)


    print("")
    print(tabulate(
        [["test", testloss, testiou, testbiou, testscore]],
        headers=["Type", "Loss", "IoU", "BIoU", "Score"]))




import os
import wandb
LOG_WANDB = False
if LOG_WANDB:
    wandb.init(project="MapAi-train")
wandb.config = {
    "epochs": 60,
    "learning_rate": 5e-5,
    "batch_size": 1,
    "task": 1,
    "augmented_data_use_ratio": 0.15,
    "augmented_image_duplications": 1,
    "include_test_dataset":True
}

DATA_FOLDER_DICT = {
    "original_images": "../../data/train/images/",
    "original_masks": "../../data/train/masks/",
    "original_images_test": "../../data/validation/images/",
    "original_masks_test": "../../data/validation/masks/",
    "include_test_dataset": wandb.config["include_test_dataset"],
    "generated_data_folders": ["1-5_inpainting_rooftops/images", "generated_dataset_1-5"],
    "generated_data_root": "/home/kaborg15/Stable_Diffusion/MapAI_Generated_images/",
    "augment_data_mode": "append", # Original, append, replace
    "augmented_data_use_ratio": wandb.config["augmented_data_use_ratio"],
    "augmented_image_duplications": wandb.config["augmented_image_duplications"]
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Training a segmentation model")

    parser.add_argument("--epochs", type=int, default=wandb.config["epochs"], help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=wandb.config["learning_rate"], help="Learning rate used during training")
    parser.add_argument("--config", type=str, default="team_morty/src/config/data.yaml", help="Configuration file to be used")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task", type=int, default=wandb.config["task"])
    parser.add_argument("--data_ratio", type=float, default=1.0,
                        help="Percentage of the whole dataset that is used")
    parser.add_argument("--resume", type=str, default="team_morty/src/model_checkpoints/best_task1_2_fresh_run.pt", help="Path to state dict to resume training from, default is None")
    #parser.add_argument("--resume", type=str, default=None, help="Path to state dict to resume training from, default is None")
    parser.add_argument("--datasets", type=dict, default=DATA_FOLDER_DICT, help="Path to original dataset")
    parser.add_argument("--batch_size", type=int, default=wandb.config["batch_size"], help="Batch size used during training")
    parser.add_argument("--prefix", type=str, default="", help="Prefix for run name")

    args = parser.parse_args()
    # Import config
    opts = load(open(args.config, "r"), Loader)

    # Combine args and opts in single dict
    try:
        opts = opts | vars(args)
    except Exception as e:
        opts = {**opts, **vars(args)}
    
    print("Overriding config batch size with wandb config batch size")
    opts[f"task{opts['task']}"]["batchsize"] = wandb.config["batch_size"]
    print("Running on device:", opts["device"])
    print("Opts:", opts)

    rundir = create_run_dir(opts)
    opts["rundir"] = rundir
    if LOG_WANDB:
        wandb.run.name = f"{opts['prefix']}-{rundir.split('/')[-1]}-{opts['lr']}-{wandb.config['batch_size']}"
    dump(opts, open(os.path.join(rundir, "opts.yaml"), "w"), Dumper)

    train(opts)
