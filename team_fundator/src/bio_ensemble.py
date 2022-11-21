import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import os
from tqdm import tqdm
import cv2
from utils import post_process_mask
import random
from competition_toolkit.eval_functions import iou, biou

def normalize(arr):
    arr = np.clip(arr, 0.0, None)
    s = np.sum(arr)
    return arr / s if s != 0 else arr 

def prob(p):
    return random.random() <= p

def swap_shuffle(arr):
    for i in range(len(arr) - 1, 0, -1):
        j = random.randint(0, i - 1)
        arr[[i, j]] = arr[[j, i]]
    return arr

class SegDataset(Dataset):

    def __init__(self, ext=".tif"):
        
        self.weights = None
        print("Loading ensemble predictions")
        self.pred = [np.load(name) for name in glob.glob( "data/ensemble_preds/task_2/*.npy")]

        print("Loading ground truths")
        self.masks = [cv2.imread(name, cv2.IMREAD_GRAYSCALE) for name in glob.glob("../../data/validation/masks/*" + ext)]

    def __len__(self):
        return self.weights.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        weights = self.weights[idx]
        scores = np.zeros(len(self.pred))

        for i, (pred, mask) in enumerate(zip(self.pred, self.masks)):
            pred = np.multiply(pred, weights)
            pred = np.sum(pred, axis=0)
            pred = np.rint(pred).astype(np.uint8)

            pred = post_process_mask(pred)

            ind_iou = iou(pred, mask)
            ind_biou = biou(pred, mask)
            scores[i] = (ind_iou + ind_biou) / 2

        sample = {"score": np.mean(scores)}

        return sample

    def set_weights(self, arr):
        self.weights = np.expand_dims(arr, (-1, -2))

NUM_WORKERS=4
POP_SIZE = 100
IND_SIZE = 6
GENS = 30

INIT_DROP_PROB = 0.1
INIT_2X_PROB = 0.2
MUTATION_PROB = 0.8
CROSSOVER_FRAC = 0.8
N_BEST = 1

BATCH_SIZE = 1

if __name__ == "__main__":
    pass

def fitness(pop, dataloader):
    dataloader.dataset.set_weights(pop)
    fitnesses = np.zeros(pop.shape[0])
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Calculating fitness"):
        fitnesses[i] = batch["score"]
    return fitnesses

def bump_mutation(ind):
    k = random.randint(0, IND_SIZE- 1)
    ind[k] *= 0.9 if prob(0.5) else 1.1
    return ind

def noise_mutation(ind):
    if prob(0.5):
        ind += np.random.normal(scale=0.04, size=IND_SIZE)
        return ind
    k = random.randint(0, IND_SIZE- 1)
    ind[k] += 0.05 * random.random() - 0.1
    return ind


def dist(ind1, ind2):
    return np.linalg.norm(ind1 - ind2, ord=1)

def crossover(p1, p2):
    c_index = random.randint(1, IND_SIZE - 2)

    c1 = p1.copy()
    c1[c_index:] = p2[c_index:].copy()

    c2 = p2.copy()
    c2[c_index:] = p1[c_index:].copy()
    
    c1 = mutate(c1)
    c2 = mutate(c2)

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
    c_fit = fitness(crowd[1], dataloader)
    crowd_fit = np.array([p_fitnesses, c_fit])
    indeces = np.argmax(crowd_fit, axis=0)
    
    return crowd[indeces, np.arange(POP_SIZE // 2)], crowd_fit[indeces, np.arange(POP_SIZE // 2)]



MUTATIONS = [bump_mutation, noise_mutation]
def mutate(ind):
    if not prob(MUTATION_PROB):
        return normalize(ind)
    for i, mutation in enumerate(MUTATIONS):
        if not prob((i + 1)/len(MUTATIONS)):
            continue
        return normalize(mutation(ind))

def init_pop():
    pop = np.zeros((POP_SIZE, IND_SIZE))

    for i in range(POP_SIZE):
        ind = np.ones(IND_SIZE)
        ind += np.random.normal(scale=0.1, size=IND_SIZE)
        if prob(INIT_DROP_PROB):
            k = random.randint(0, IND_SIZE - 1)
            ind[k] = 0
        if prob(INIT_2X_PROB):
            k = random.randint(0, IND_SIZE - 1)
            ind[k] *= 1 + 0.3 * random.random()
        ind = normalize(ind)
        pop[i] = ind
    return pop

def main():
    dataset = SegDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    pop = init_pop()
    fit = fitness(pop, dataloader)
    ids = np.arange(POP_SIZE)
    for gen in range(GENS):

        print(f"Gen {gen + 1} of {GENS}")
        np.random.shuffle(ids)
        new_pop = np.zeros_like(pop)
        new_fit = np.zeros_like(fit)
        parent1_ids, parent2_ids = np.split(ids, 2)
        #parent2_ids = swap_shuffle(parent1_ids.copy())

        crowds_1 = np.zeros((2, POP_SIZE // 2, IND_SIZE))
        crowds_2 = np.zeros((2, POP_SIZE // 2, IND_SIZE))

        for i, (p1_id, p2_id) in enumerate(zip(parent1_ids, parent2_ids)):
            p1, p2 = pop[p1_id], pop[p2_id]

            if prob(CROSSOVER_FRAC):
                cr1, cr2 = crossover(p1, p2)
            else:
                cr1, cr2 = np.array([p1, mutate(p1)]), np.array([p2, mutate(p2)])

            crowds_1[:, i] = cr1
            crowds_2[:, i] = cr2

        crowds_1, fit1 = uncrowd(np.array(crowds_1), fit[parent1_ids], dataloader)
        crowds_2, fit2 = uncrowd(np.array(crowds_2), fit[parent2_ids], dataloader)

        new_pop[parent1_ids] = crowds_1
        new_pop[parent2_ids] = crowds_2

        new_fit[parent1_ids] = fit1
        new_fit[parent2_ids] = fit2

        n_worst_new = np.argpartition(new_fit, N_BEST)[:N_BEST]
        n_best_old = np.argpartition(fit, -N_BEST)[-N_BEST:]

        new_fit[n_worst_new] = fit[n_best_old]
        new_pop[n_worst_new] = pop[n_best_old]

        fit = new_fit
        pop = new_pop

        print("Current best fit: ", np.max(fit), pop[np.argmax(fit)])
    
    print(pop[n_worst_new])
    print(fit[n_worst_new])

if __name__ == "__main__":
    main()
