from mimetypes import init
import os
import glob
import torch
import segmentation_models_pytorch as smp
from optimizers import PolyLR, RAdam, AdamWarmup
from torch.optim.lr_scheduler import MultiStepLR
from yaml import load, Loader
import cv2
import numpy as np

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
    """
    Returns an initialized optimizer for the provided model based on the values in opts
    """
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
    "SoftCrossEntropyLoss": smp.losses.SoftCrossEntropyLoss
}

def get_losses(opts):
    """
    Returns a weighted multiloss function based on the values in opts
    """
    losses_cfg = opts["training"]["losses"]

    used_losses = []
    weights = torch.tensor(losses_cfg["weights"])
    for loss_name in losses_cfg["names"]:
        init_params = losses_cfg[loss_name]['init_params'] if losses_cfg[loss_name]['init_params'] is not None else {}

        if "mode" in init_params:
            init_params["mode"] = "multiclass" if opts["num_classes"] > 1 else "binary"
        
        if loss_name == "CrossEntropy":
            loss_name = "SoftCrossEntropyLoss" if opts["num_classes"] > 1 else "SoftBCEWithLogitsLoss"
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
    """
    Returns an smp model based on the values in opts and the current task
    """
    model_cfg = opts["model"]
    model = None
    if "in_channels" not in model_cfg:
        model_cfg["in_channels"] = 4 if int(opts["task"]) == 2 else 3
    if opts["task"] == 3:
        model_cfg["in_channels"] = 1

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
    """
    Returns an initialized learning rate scheduler
    """
    schedule_cfg = opts["training"]["scheduler"]
    scheduler = schedule_cfg.get("name", "PolyLR")

    init_params = schedule_cfg.get("init_params", {"epochs": opts["train"]["epochs"]} if scheduler == "PolyLR" else {"milestones": [int(opts["train"]["epochs"] * 0.8)]})

    return schedules[scheduler](optimizer, **init_params)

def get_aug_names(opts, augmentation_cfg, transforms):
    """
    Returns the augmentation schedule for all the epochs during training
    """
    aug_list = []
    if opts["task"] > 2:
        return ["task3_and_4_augs"] * opts["train"]["epochs"]
        
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


data_configs = {
    "mapai": "mapai.yaml",
    "landcover": "landcover.yaml",
    "mapai_lidar_masks": "mapai_lidar_masks.yaml",
    "mapai_reclassified": "mapai_reclassified.yaml",
}

def merge_dict(base, extension):
    """
    Merges two dicts, including any dicts within them 
    """
    for k, v in extension.items():
        if type(v) == dict:
            v =  merge_dict(base[k], v)
        base[k] = v
    return base


def get_dataset_config(opts):
    """
    Returns the config assosciated with the dataset selected in opts
    """
    dataset = opts["dataset"]
    base_opts = load(open("config/datasets/base_dataset.yaml", "r"), Loader)
    dataset_opts = load(open(f"config/datasets/{data_configs[dataset]}", "r"), Loader)

    dataset_opts = merge_dict(base_opts, dataset_opts)
    return dataset_opts

def post_process_mask(pred: np.ndarray) -> np.ndarray:
    min_total_area = 1500
    fill_threshold = 10

    remove_treshhold = 180
    max_edge_ratio = 0.05


    contours, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        hierarchy = [[]]

    total_area =  np.sum(pred)
    for i, (c, h) in enumerate(zip(contours, hierarchy[0])):
        _, _, _, parent = h
        area = cv2.contourArea(c)

        if parent >= 0:
            if area <= fill_threshold:
                cv2.drawContours(pred, contours, i, color=1, thickness=-1)
            continue

        if area > remove_treshhold or total_area < min_total_area:
            continue

        edges = 0
        for t in c:
            y = t[0][0]
            x = t[0][1]
            if y == 0 or x == 0 or y == 499 or x == 499:
                edges += 1
        
        if edges == 0 or area == 0 or edges / area < max_edge_ratio:
            cv2.drawContours(pred, contours, i, color=0, thickness=-1)
    
    return pred
