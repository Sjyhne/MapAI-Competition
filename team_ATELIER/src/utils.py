import os
import glob
import torch
import onnxruntime as ort
from pathlib import Path
from typing import Callable, List, Union
import numpy as np
from skimage.morphology import remove_small_objects, remove_small_holes


def cleanup_image(image, hole_size = 150, object_size=50):
    positive = (image[0,0,...]< image[0,1,...]).astype(np.uint8)
    padded = np.pad(positive, 1, constant_values = 1)
    hole_removed = remove_small_holes(padded, hole_size)[1:-1,1:-1]

    padded = np.pad(hole_removed, 1, constant_values = 0)
    filtered = remove_small_objects(padded, object_size)[1:-1,1:-1]

    img=np.stack([1-filtered, filtered], axis=0).astype(np.float32)
    return img.reshape((1,)+img.shape)


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


def create_model(model_fpath: Path, cpu=False):
    if cpu:
        ort_session = ort.InferenceSession(
            str(model_fpath),
            providers=["CPUExecutionProvider"],
        )
    else:
        ort_session = ort.InferenceSession(
            str(model_fpath),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )


    def model(x: Union[torch.FloatTensor, np.ndarray]):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        return ort_session.run(None, {"feature": x})[0]

    return model


def predict(feature: torch.FloatTensor, models: Union[List[Callable], Callable], numpy=False):
    predictions = []
    models = [models] if isinstance(models, Callable) else models

    for model in models:
        if feature.shape[1] == 3:
            if feature.max() <= 1.0:
                scale = np.array([255.,255.,255.]).reshape(1,3,1,1).astype(np.float32)
                predictions.append(model(feature*255.0))
            else:
                predictions.append(model(feature))
        else:
            if feature[:,0:3,...].max() <= 1.0:
                scale = np.array([255.,255.,255.,1.0]).reshape(1,4,1,1).astype(np.float32)
                predictions.append(model(feature*scale))
            else:
                predictions.append(model(feature))

    if len(models) > 1:
        prediction = np.stack(predictions).mean(axis=0)
    else:
        prediction = predictions[0]

    prediction = cleanup_image(prediction)

    if numpy:
        return prediction
    else:
        return torch.from_numpy(prediction)
