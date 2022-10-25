import os
import glob
import torch
import segmentation_models_pytorch as smp

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

def get_optimizer(opts, model):
    optimizer_cfg = opts["training"]["optimizer"]
    if optimizer_cfg["name"] == "Adam":
        return torch.optim.Adam(model.parameters(), lr=optimizer_cfg["lr"])
    elif optimizer_cfg["name"] == "SGD":
        return torch.optim.SGD(model.parameters(), lr=optimizer_cfg["lr"])
    else:
        print(f"Optimizer {optimizer_cfg['name']} is not implemented")
        exit()

# TODO: test and use
def get_losses(opts):
    losses_cfg = opts["training"]["losses"]

    losses = {
        "DiceLoss": smp.losses.DiceLoss,
        "JaccardLoss": smp.losses.JaccardLoss,
        "TverskyLoss": smp.losses.TverskyLoss,
        "FocalLoss": smp.losses.FocalLoss,
        "LovaszLoss": smp.losses.LovaszLoss,
        "SoftBCEWithLogitsLoss": smp.losses.SoftBCEWithLogitsLoss,
        "SoftCrossEntropyLoss": smp.losses.SoftCrossEntropyLoss
    }
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
        "DeepLabV3": smp.DeepLabV3Plus
    }

def get_model(opts):
    model_cfg = opts["model"]
    model = None
    if "in_channels" not in model_cfg:
        model_cfg["in_channels"] = 3 if int(opts["task"]) == 1 else 4

    if model_cfg["name"] in smp_models.keys():
        model = smp_models[model_cfg["name"]](
            encoder_name=model_cfg.get("encoder", "resnet34"),        
            encoder_weights=model_cfg.get("encoder_weights", "imagenet"),    
            in_channels=model_cfg["in_channels"],
            classes=opts.get("num_classes", 2),
            encoder_depth=model_cfg.get("encoder_depth", 5)
        )
    else:
        print(f"Model {model_cfg['name']} is not available")
        exit()
    return model