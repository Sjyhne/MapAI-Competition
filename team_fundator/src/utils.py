from mimetypes import init
import os
import glob
import torch
import segmentation_models_pytorch as smp
from optimizers import PolyLR, RAdam, AdamWarmup
from torch.optim.lr_scheduler import MultiStepLR

def create_run_dir(opts, dataset_dir=""):

    rundir = "runs"

    rundir = os.path.join(rundir, "task_" + str(opts["task"]))
    if dataset_dir != "":
        rundir = os.path.join(rundir, dataset_dir)

    if not os.path.exists(rundir):
        os.makedirs(rundir, exist_ok=True)

    existing_folders = os.listdir(rundir)

    if len(existing_folders) == 0:
        curr_run_dir = "run_0"
    else:
        runs = []
        for folder in existing_folders:
            _, number = folder.split("_")
            runs.append(int(number))

        curr_run_dir = "run_" + str(max(runs) + 1)

    runpath = os.path.join(rundir, curr_run_dir)

    os.mkdir(runpath)

    return runpath

def store_model_weights(opts: dict, model: torch.nn.Module, type: str, epoch: int):
    rundir = opts["rundir"]
    files = glob.glob(os.path.join(rundir, f"{type}_*.pt"))
    for f in files:
        os.remove(f)
    torch.save(model.state_dict(), os.path.join(rundir, f"{type}_task{opts['task']}_{epoch}.pt"))

def record_scores(opts, scoredict):
    rundir = opts["rundir"]

    with open(os.path.join(rundir, "run.log"), "a") as f:
        f.write(str(scoredict) + "\n")

optimizers = {
    "Adam": torch.optim.Adam,
    "RAdam": RAdam,
    "AdamWarmup": AdamWarmup,
    "SGD": torch.optim.SGD
}

def get_optimizer(opts, model):
    optimizer_cfg = opts["training"]["optimizer"]
    if optimizer_cfg["name"] not in optimizers:
        print(f"Optimizer {optimizer_cfg['name']} is not implemented")
        exit()
    return optimizers[optimizer_cfg["name"]](model.parameters(), **optimizer_cfg["init_params"] )

losses = {
    "DiceLoss": smp.losses.DiceLoss,
    "JaccardLoss": smp.losses.JaccardLoss,
    "TverskyLoss": smp.losses.TverskyLoss,
    "FocalLoss": smp.losses.FocalLoss,
    "LovaszLoss": smp.losses.LovaszLoss,
    "SoftBCEWithLogitsLoss": smp.losses.SoftBCEWithLogitsLoss,
    "SoftCrossEntropyLoss": smp.losses.SoftCrossEntropyLoss,
    "MCCLoss": smp.losses.MCCLoss
}

def get_losses(opts):
    losses_cfg = opts["training"]["losses"]

    used_losses = []
    weights = torch.tensor(losses_cfg["weights"])
    for loss_name in losses_cfg["names"]:
        init_params = losses_cfg[loss_name]['init_params'] if losses_cfg[loss_name]['init_params'] is not None else {}
        used_losses.append(losses[loss_name](**init_params))

    def multiloss(preds, targets):
        loss = 0
        for i in range(len(used_losses)):
            loss += weights[i] * used_losses[i](preds, targets)
        return loss

    return multiloss


smp_models = {
        "UNet": smp.Unet,
        "UNet++": smp.UnetPlusPlus,
        "MAnet": smp.MAnet,
        "Linknet": smp.Linknet,
        "FPN": smp.FPN,
        "DeepLabV3": smp.DeepLabV3,
        "DeepLabV3+": smp.DeepLabV3Plus
    }

def get_model(opts):
    model_cfg = opts["model"]
    model = None
    if "in_channels" not in model_cfg:
        model_cfg["in_channels"] = 4 if int(opts["task"]) == 2 else 3
    if opts["task"] == 3:
        model_cfg["aux_head"] = False

    aux_params = model_cfg["aux_head_params"] if model_cfg["aux_head"] else None

    if model_cfg["name"] in smp_models.keys():
        model = smp_models[model_cfg["name"]](
            encoder_name=model_cfg.get("encoder", "resnet34"),        
            encoder_weights=model_cfg.get("encoder_weights", "imagenet"),    
            in_channels=model_cfg["in_channels"],
            classes=opts["num_classes"],
            encoder_depth=model_cfg.get("encoder_depth", 5),
            aux_params=aux_params
        )
    else:
        print(f"Model {model_cfg['name']} is not available")
        exit()
    return model


schedules = {
    "PolyLR": PolyLR,
    "MultiStep": MultiStepLR,
}



def get_scheduler(opts, optimizer):
    schedule_cfg = opts["training"]["scheduler"]
    scheduler = schedule_cfg.get("name", "PolyLR")

    init_params = schedule_cfg.get("init_params", {"epochs": opts["epochs"]} if scheduler == "PolyLR" else {"milestones": [int(opts["epochs"] * 0.8)]})

    return schedules[scheduler](optimizer, **init_params)

def get_aug_names(opts, augmentation_cfg, transforms):
    aug_list = []
    for i in range(opts["train"]["epochs"]):
        if i >= augmentation_cfg["warmup_epochs"]:
            aug = augmentation_cfg["cycle"][(i - augmentation_cfg["warmup_epochs"]) % len(augmentation_cfg["cycle"])]
            if aug not in transforms:
                print(f"Unsupported transform {aug}")
                exit()
                
            aug_list.append(aug)
            continue
        aug_list.append(augmentation_cfg.get("initial", "normal"))
    return aug_list