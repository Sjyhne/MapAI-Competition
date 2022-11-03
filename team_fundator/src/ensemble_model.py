import glob
from pathlib import Path
from typing import List, Union
import torch
import yaml

from utils import get_model


class EnsembleModel(torch.nn.Module):
    """Ensemble of torch models, pass tensor through all models and average results"""

    def __init__(self, models: list):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = None
        model_preds = []
        for model in self.models:
            y = model(x)
            model_preds.append(y)  # .to("cpu"))
            if result is None:
                result = y
            else:
                result += y
        result /= torch.tensor(len(self.models)).to(result.device)
        return {"result": result, "model_preds": model_preds}


def load_models_from_runs(
    run_folder: str, run_numbers: Union[str, List[int]]
) -> List[torch.nn.Module]:
    if isinstance(run_numbers, str) and run_numbers == "*":
        run_folders = [
            str(p)
            for p in Path(run_folder).iterdir()
            if p.is_dir() and p.name.startswith("run_")
        ]
    elif isinstance(run_numbers, list):
        run_folders = [f"{run_folder}/run_{run_num}" for run_num in run_numbers]
    else:
        raise ValueError(
            f'Invalid run numbers argument: {run_numbers}. Must be "*" or list of run numbers'
        )
    configs = [
        yaml.load(open(f"{run}/opts.yaml", "r"), yaml.Loader) for run in run_folders
    ]
    checkpoints = [torch.load(glob.glob(f"{run}/best*.pt")[0]) for run in run_folders]
    models = []
    for config, checkpoint in zip(configs, checkpoints):
        model = get_model(config)
        model.load_state_dict(checkpoint)
        models.append(model)
    return models, [Path(rf).name for rf in run_folders]
