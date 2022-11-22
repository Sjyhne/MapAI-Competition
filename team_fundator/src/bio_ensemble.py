import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
from utils import create_run_dir
from tqdm import tqdm
import cv2
from utils import post_process_mask
import random
from competition_toolkit.eval_functions import iou, biou
import argparse
import os

def normalize(arr):
    arr = np.clip(arr, 0.0, None)
    s = np.sum(arr)
    return arr / s if s != 0 else arr 

def prob(p):
    #helper function which returns True with probability p
    return random.random() <= p

def save_ensemble(pop: np.ndarray, fit: np.ndarray, rundir: str, epoch: int):
    files = glob.glob(os.path.join(rundir, f"last_*.npy"))
    for f in files:
        os.remove(f)
    np.save(pop, os.path.join(rundir, f"last_pop_{epoch}.npy"))
    np.save(fit, os.path.join(rundir, f"last_fit_{epoch}.npy"))

class PredDataset(Dataset):
    """
    Dataset which loads predictions and tests the weightins of the ensemble.
    Used to support multithreading
    """
    def __init__(self, args, ext=".tif"):
        
        self.weights = None
        print("Loading ensemble predictions")
        self.pred = [np.load(name) for name in glob.glob(f"data/ensemble_preds/task_{args.task}/*.npy")]

        print("Loading ground truths")
        self.masks = [cv2.imread(name, cv2.IMREAD_GRAYSCALE) for name in glob.glob("../../data/validation/masks/*" + ext)]

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

POP_SIZE = None
IND_SIZE = None # Number of weights to evolve / number of models in the ensemble
GENS = None

INIT_DROP_PROB = 0.1 # Probability of setting a weight to zero during initialisation of the population
INIT_2X_PROB = 0.2 # probability of multiplying a weight with a small factor during initialisation of the population
MUTATION_PROB = 0.85 # Probability of performaing a mutation (In crossover or otherwise)
CROSSOVER_FRAC = 0.8 # Probability of performing crossover. Other wise mutation is (probably) done


def fitness(pop, dataloader):
    dataloader.dataset.set_weights(pop)
    fitnesses = np.zeros(pop.shape[0])
    # Calculate fitness. One 'batch' is one individual
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Calculating fitness"):
        fitnesses[i] = batch["score"]
    return fitnesses

def noise_mutation(ind):
    # With equal probability:
    # Add (or subtract) some noise on the entire chromosome or a little more noise on one index
    
    scalar = 7 / IND_SIZE # scales the domain of the added noise to be appropriate for the IND SIZE

    if prob(0.5):
        ind += np.random.normal(scale=0.02 * scalar, size=IND_SIZE)
        return normalize(ind)
    k = random.randint(0, IND_SIZE - 1)
    ind[k] += scalar * 0.03 * (random.random() - 1.0)
    return normalize(ind)


def dist(ind1, ind2):
    # L1 distance for crowding
    return np.linalg.norm(ind1 - ind2, ord=1)

def crossover(p1, p2):
    c_index = random.randint(1, IND_SIZE - 1)

    #crossover
    c1 = p1.copy()
    c1[c_index:] = p2[c_index:].copy()

    c2 = p2.copy()
    c2[c_index:] = p1[c_index:].copy()
    
    #mutation
    c1 = mutate(c1)
    c2 = mutate(c2)

    #return parent, child pairs
    return crowd(p1, c1, p2, c2)

def crowd(p1, c1, p2, c2):
    # crowding
    crowd_1_dist = dist(c1, p1) + dist(c2, p2)
    crowd_2_dist = dist(c1, p2) + dist(c2, p1)
    
    if crowd_1_dist < crowd_2_dist:
        crowd_1 = np.array([p1, c1])    
        crowd_2 = np.array([p2, c2])
    else:
        crowd_1 = np.array([p1, c2])    
        crowd_2 = np.array([p2, c1])

    return crowd_1, crowd_2

def uncrowd(crowd, p_fitnesses, dataloader):
    # select individuals from parent, child pairs based on best fit
    c_fit = fitness(crowd[1], dataloader)
    crowd_fit = np.array([p_fitnesses, c_fit])
    indeces = np.argmax(crowd_fit, axis=0)
    
    return crowd[indeces, np.arange(len(indeces))], crowd_fit[indeces, np.arange(len(indeces))]


MUTATIONS = [noise_mutation]
def mutate(ind):
    # With equal probability perform a mutation from MUTATIONS
    if not prob(MUTATION_PROB):
        return normalize(ind)

    p = random.random()
    for i, mutation in enumerate(MUTATIONS):
        if p > (i + 1)/len(MUTATIONS):
            continue
        return mutation(ind)

def init_pop():
    # Initialise the population
    pop = np.zeros((POP_SIZE, IND_SIZE))

    scalar = 7 / IND_SIZE
    for i in range(POP_SIZE):
        ind = np.ones(IND_SIZE)

        ind += np.random.normal(scale=0.1 * scalar, size=IND_SIZE)
        if prob(INIT_DROP_PROB):
            k = random.randint(0, IND_SIZE - 1)
            ind[k] = 0
        if prob(INIT_2X_PROB):
            k = random.randint(0, IND_SIZE - 1)
            ind[k] *= 1 + 0.3 * random.random()
        ind = normalize(ind)
        pop[i] = ind
    return pop

def main(args, rundir):
    global POP_SIZE
    global IND_SIZE
    global GENS

    POP_SIZE = args.pop_size
    IND_SIZE = args.size
    GENS = args.gens

    dataset = PredDataset(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    
    if args.start_folder is not None:
        pop_path = glob.glob(os.path.join(args.start_folder, f"last_pop*.npy"))
        fit_path = glob.glob(os.path.join(args.start_folder, f"last_fit*.npy"))
        fit = np.load(fit_path)
        pop = np.load(pop_path)

        POP_SIZE, IND_SIZE = pop.shape
    else:
        pop = init_pop()
        fit = fitness(pop, dataloader)
    ids = np.arange(POP_SIZE)

    print("Current best fit: ", np.max(fit), pop[np.argmax(fit)], "Median fit: ", np.median(fit))
    for gen in range(GENS):
        print(f"Gen {gen + 1} of {GENS}")

        np.random.shuffle(ids)
        
        parent1_ids, parent2_ids = np.split(ids, 2)

        crowds = np.zeros((2, POP_SIZE, IND_SIZE))

        # get next generation
        for i, (p1_id, p2_id) in enumerate(zip(parent1_ids, parent2_ids)):
            p1, p2 = pop[p1_id], pop[p2_id]

            if prob(CROSSOVER_FRAC):
                cr1, cr2 = crossover(p1, p2)
            else:
                cr1, cr2 = crowd(p1, noise_mutation(p1.copy()), p2, noise_mutation(p2.copy()))

            crowds[:, i] = cr1
            crowds[:, i + POP_SIZE // 2] = cr2

        # finish crowding parent, child pairs
        next_pop, next_fit = uncrowd(crowds, fit[ids], dataloader)

        fit = next_fit
        pop = next_pop

        print("Current best fit: ", np.max(fit), pop[np.argmax(fit)], "Median fit: ", np.median(fit))

        save_ensemble(pop, fit, rundir, gen)

    print(pop[np.argmax(fit)])
    print(fit[np.argmax(fit)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=1, help="Which task you are testing")
    parser.add_argument("--size", type=int, default=4, help="How many ensembles are you testing?")
    parser.add_argument("--workers", type=int, default=4, help="How many workers do you want?")
    parser.add_argument("--pop-size", type=int, default=100, help="How large population do you want?")
    parser.add_argument("--gens", type=int, default=30, help="How many generations?")
    parser.add_argument("--start-folder", type=str, default=None, help="Folder to load saved population")


    args = parser.parse_args()

    # task 1: [0.25023021 0.25033406 0.24972593 0.2497098 ]
    # task 2:

    rundir = create_run_dir({"task": "bio"})

    main(args, rundir)
