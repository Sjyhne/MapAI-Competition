import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
import cv2
from utils import post_process_mask
from competition_toolkit.eval_functions import iou, biou
import argparse


class PredDataset(Dataset):
    """
    Dataset which loads predictions and tests the weightins of the ensemble.
    Used to support multithreading
    """
    def __init__(self, args, ext=".tif"):
        
        self.weights = None
        print("Loading ensemble predictions")
        self.pred = [np.load(name) for name in sorted(glob.glob(f"data/ensemble_preds/task_{args.task}{'_tta' if args.tta else ''}/*.npy"))]

        print("Loading ground truths")
        self.masks = [cv2.imread(name, cv2.IMREAD_GRAYSCALE) for name in sorted(glob.glob("../../data/validation/masks/*" + ext))]

    def __len__(self):
        # The number of different weights to test. Matches POP_SIZE in the EA
        return self.weights.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        weights = self.weights[idx]
        scores = np.zeros(len(self.pred))

        # get the score for the weights at index idx
        for i, (pred, mask) in enumerate(zip(self.pred, self.masks)):
            pred = np.multiply(pred, weights)
            pred = np.sum(pred, axis=0)
            pred = np.rint(pred).astype(np.uint8)

            pred = post_process_mask(pred)

            ind_iou = iou(pred, mask)
            ind_biou = biou(pred, mask)
            scores[i] = (ind_iou + ind_biou) / 2

        # return mean score
        sample = {"score": np.mean(scores)}

        return sample

    def set_weights(self, arr):
        self.weights = np.expand_dims(arr, (-1, -2))

def fitness(pop, dataloader):
    dataloader.dataset.set_weights(pop)
    fitnesses = np.zeros(pop.shape[0])
    # Calculate fitness. One 'batch' is one individual
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Calculating fitness"):
        fitnesses[i] = batch["score"]
    return fitnesses


def get_weights(size: int, min_models: int, max_models: int):
    weights = [list(map(int, np.binary_repr(num, width=size))) for num in range(2 ** size)]
    weights = list(filter(lambda x: sum(x) >= min_models and sum(x) <= max_models, weights))
    weights = np.array(weights)
    s = np.sum(weights, axis=1)
    return weights / s[:, None]

def main(args):

    dataset = PredDataset(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    
    w = get_weights(args.size, args.min_ensemble_size, args.max_ensemble_size)
    f = fitness(w, dataloader)

    best_idx = np.argpartition(f, range(-5, 0))[-5:]
    for (weight, fit) in zip(w[best_idx], f[best_idx]):
        print(np.uint8(weight > 0), fit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=1, help="Which task you are testing")
    parser.add_argument("--workers", type=int, default=6, help="How many workers do you want?")
    parser.add_argument("--size", type=int, default=9, help="How many ensembles are you testing?")
    parser.add_argument("--min-ensemble-size", type=int, default=6, help="The minimum number of models in each ensemble?")
    parser.add_argument("--max-ensemble-size", type=int, default=10000000, help="The maximum number of models in each ensemble?")
    parser.add_argument("--tta", action="store_true", help="Whether to perform tta with rotation during inference")


    args = parser.parse_args()

    # task 1: [0.25023021 0.25033406 0.24972593 0.2497098 ]
    # task 2:


    main(args)
