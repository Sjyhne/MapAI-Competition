import glob
from pathlib import Path
from typing import List, Union
import torch
import yaml

from utils import get_model

class EnsembleModel(torch.nn.Module):
    """Ensemble of torch models, pass tensor through all models and average results"""

    def __init__(self, models: list, sum_outputs):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.sum_outputs = sum_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = None
        model_preds = []
        for model in self.models:
            in_channels = next(model.parameters()).shape[1]
            if in_channels == x.shape[1]:
                y = model(x)
            elif in_channels == 1:
                y = model(x[:, -1])
            elif in_channels == 3:
                y = model(x[:, 0:3])

            if len(y) == 2:
                y, aux_label = y
            y = y.to("cpu")
            model_preds.append(y)
            if self.sum_outputs:
                if result is None:
                    result = y
                    continue
                result += y

        if self.sum_outputs:
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
        config =  load(open(config, "r"), Loader)
        model = get_model(config)
        model.load_state_dict(torch.load(checkpoint[0]))
        models.append(model)
    return models, [Path(rf).name for rf in run_folders]
