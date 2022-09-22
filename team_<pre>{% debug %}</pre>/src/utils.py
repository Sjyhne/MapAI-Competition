import os
import glob
import torch

def create_run_dir(opts):

    rundir = "runs"

    rundir = os.path.join(rundir, "task_" + str(opts["task"]))

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