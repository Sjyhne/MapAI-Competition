import yaml
import pandas as pd
import numpy as np
from pathlib import Path


def load_run_log(logfile: str):
    with open(logfile, "r") as f:
        log_lines = [s.strip() for s in f.readlines()]
    yml_dicts = [yaml.safe_load(line) for line in log_lines]
    out = {
        k: []
        for k in [
            "epoch",
            "trainloss",
            "testloss",
            "trainiou",
            "testiou",
            "trainbiou",
            "testbiou",
            "trainscore",
            "testscore",
        ]
    }
    for d in yml_dicts:
        for k, v in d.items():
            out[k].append(v)
    return out


def load_run_config(config_file: str):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def create_model_leaderboard(runs):
    result_dict = {
        k: []
        for k in [
            "Run#",
            "ModelType",
            "Backbone",
            "Epochs",
            "TrainBestEpoch",
            "TrainIoU",
            "TrainBIoU",
            "TrainScore",
            "ValBestEpoch",
            "ValIoU",
            "ValBIoU",
            "ValScore",
        ]
    }
    for run in runs:
        run_config = load_run_config(run.joinpath("opts.yaml"))
        run_log = load_run_log(run.joinpath("run.log"))
        result_dict["Run#"].append(run.name)
        result_dict["ModelType"].append(run_config["model"]["name"])
        result_dict["Backbone"].append(run_config["model"]["encoder"])
        result_dict["Epochs"].append(len(run_log["epoch"]))
        result_dict["TrainBestEpoch"].append(
            run_log["epoch"][np.argmax(run_log["trainscore"])]
        )
        result_dict["TrainIoU"].append(max(run_log["trainiou"]))
        result_dict["TrainBIoU"].append(max(run_log["trainbiou"]))
        result_dict["TrainScore"].append(max(run_log["trainscore"]))
        result_dict["ValBestEpoch"].append(
            run_log["epoch"][np.argmax(run_log["testscore"])]
        )
        result_dict["ValIoU"].append(max(run_log["testiou"]))
        result_dict["ValBIoU"].append(max(run_log["testbiou"]))
        result_dict["ValScore"].append(max(run_log["testscore"]))
    return pd.DataFrame(result_dict).set_index("Run#").sort_values(by="ValScore")


def get_filtered_runs(runs_dir):
    # Get all runs containing logs
    runs = [run for run in Path(runs_dir).iterdir() if run.name.startswith("run_")]
    filtered_runs = [run for run in runs if Path(run).joinpath("run.log").exists()]
    print(f"Found {len(runs)} runs - {len(filtered_runs)} containing logs:")
    print([run.name for run in filtered_runs])
    return filtered_runs


def create_long_result_df(runs):
    # Collect all run logs in wide dataframe
    df = pd.DataFrame()
    for run in runs:
        log = load_run_log(run.joinpath("run.log"))
        log["Run#"] = [run.name] * len(log["epoch"])
        df = pd.concat([df, pd.DataFrame(log)])

    # Create long-form dataframe from wide df (suited for seaborn plotting)
    melted = df.melt(id_vars=["Run#", "epoch"])
    melted["set"] = ["test" if "test" in s else "train" for s in melted["variable"]]
    melted["variable"] = melted["variable"].apply(
        lambda s: s[5:] if "train" in s else s[4:]
    )
    return melted
