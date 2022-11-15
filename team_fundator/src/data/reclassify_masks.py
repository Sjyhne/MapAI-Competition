import glob
import numpy as np
import cv2
from collections import defaultdict
from math import  sqrt

from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TreeNode():
    """
    Implementation of a disjoint set. Used to group pixels belonging to the same building
    """
    def __init__(self, root=None) -> None:
        self.root: TreeNode = root if root is not None else self
        self.rank = 0

    
    def get_root(self):
        if self.root == self:
            return self
        self.root = self.root.get_root()
        return self.root

    def link_trees(self, node2):
        if self.rank > node2.rank:
            node2.root = self
        else:
            self.root = node2
            if (self.rank == node2.rank):
                node2.rank += 1

    def joinTrees(self, node2):
        if self.get_root() != node2.get_root():
            self.root.link_trees(node2.root)        


def get_nswe_neighbours(y:  int, x: int, ymax: int, xmax: int):
    nbs = []
    if y - 1 >= 0:
        nbs.append((y - 1, x))
    if x - 1 >= 0:
        nbs.append((y, x - 1))
    if y + 1 < ymax:
        nbs.append((y + 1, x))
    if x + 1 < xmax:
        nbs.append((y, x + 1))
    return nbs

def get_all_neighbours(y: int, x: int, ymax: int, xmax: int):
    nbs = []
    for ny in range(max(0, y - 1), min(y + 2, ymax)):
        for nx in range(max(0, x - 1), min(x + 2, xmax)):
            if ny == y and nx == x:
                continue
            nbs.append((ny, nx))
    return nbs

def l2(y_diff: int, x_diff: int):
    return sqrt(y_diff**2 + x_diff**2)
    #return abs(y_diff) + abs(x_diff)

def find_buildings(grid: np.ndarray):
    forest = defaultdict(lambda: defaultdict(lambda: TreeNode()))
    edges_list = set()
    
    #find all pixels on the edge of a building, i.e. pixels neighbouring an empty pixel, or the edge of the image
    #also join pixels which are labelled building with the disjoint set class
    for y in range(grid.shape[0] - 1):
        for x in range(grid.shape[1] - 1):
            if grid[y][x] != grid[y][x + 1]:
                offset = grid[y][x + 1]
                edges_list.add((y, x + offset))
            elif grid[y][x] == 1:
                node = forest[y][x]
                node.joinTrees(forest[y][x + 1])
                if y == 0 or x == 0:
                    edges_list.add((y, x))

            if grid[y][x] != grid[y + 1][x]:
                offset = grid[y + 1][x]
                edges_list.add((y + offset, x))
            elif grid[y][x] == 1:
                node = forest[y][x]
                node.joinTrees(forest[y + 1][x])
                if y == 0 or x == 0:
                    edges_list.add((y, x))
        x += 1
        if grid[y][x] == 1:
            edges_list.add((y, x))
            if y + 1 < grid.shape[0] and grid[y + 1][x] == 1:
                node = forest[y][x]
                node.joinTrees(forest[y + 1][x])
    y += 1
    for x in range(grid.shape[1] - 1):
        if grid[y][x] == 1:
            edges_list.add((y, x))
            if x + 1 < grid.shape[1] and grid[y][x + 1] == 1:
                node = forest[y][x]
                node.joinTrees(forest[y][x + 1])
    if grid[grid.shape[0] - 1][grid.shape[1] - 1] == 1:
        edges_list.add((grid.shape[0] - 1, grid.shape[1] - 1))

    return forest, edges_list

class gtDataset(Dataset):
    """
    A torch dataset to support multithreading of the reclassification code
    """
    def __init__(self, dir: str, max_dist: float, min_building_size: int):
        self.max_dist = max_dist
        self.images = glob.glob(dir)
        self.min_building_size = min_building_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        full_grid = cv2.imread(self.images[idx], cv2.IMREAD_UNCHANGED)
        grid = full_grid.copy()
        full_grid = np.stack([grid, grid, grid], axis=-1)

        assert np.max(grid) <= 1
        forest, edges_list = find_buildings(grid)

        if len(edges_list) == 0: # no buildings in the image
            return {'gt': full_grid, "path": self.images[idx]}

        classified_building_edges = np.zeros(grid.shape[:2], dtype=np.uint8)

        first_fronts = defaultdict(lambda: set()) # stores the current front in the distance search.
        closest_from_tree = defaultdict(lambda: defaultdict(lambda: set())) # stores the n closest edge pixels from a building for each pixel
        # for each pixel, it denotes if the iterative search from a given building has already visited the pixel
        visited_by_tree = defaultdict(lambda: defaultdict(lambda: False)) 
        tree_circumferences = defaultdict(lambda: 0) # counts the number of pixels in edges_list from each building

        all_dists = defaultdict(lambda: np.ones(grid.shape) * self.max_dist)

        # perform an iterative search starting in the edge of a building to find the closest edge pixel for each pixel reached by the search.
        # based on the intuition that for a given pixel its closest edge pixel must be the closest edge pixel of one of its neighbours

        # each iteration is based on the current front, which is given by the unvisited neigbours of the previous front:
        #      f_1       f_2           f_3     
        #                           # # # # #
        #               # # #       #       #
        #       #       #   #       #       #          ...
        #               # # #       #       #
        #                           # # # # #
        # this is repeated for every building

        # generate the first front, and label the edge class for the final mask
        for y, x in edges_list:
            classified_building_edges[y][x] = 1
            root = forest[y][x].get_root()
            tree_circumferences[root] += 1
            nbs = get_all_neighbours(y, x, grid.shape[0], grid.shape[1]) 
            for ny, nx in nbs:
                if grid[ny][nx] == 1:
                    classified_building_edges[ny][nx] = 1
                    continue

                dist = l2(ny - y, nx - x)
                if dist < all_dists[root][ny][nx]:
                    all_dists[root][ny][nx] = dist
                    closest_from_tree[root][(ny, nx)] = {(y, x)} 
                    visited_by_tree[root][(ny, nx)] = True
                elif dist == all_dists[root][ny][nx]:
                    closest_from_tree[root][(ny, nx)].add((y, x))
                for nny, nnx in get_all_neighbours(ny, nx, grid.shape[0], grid.shape[1]):
                    if grid[nny][nnx] != 0:
                        continue
                    first_fronts[root].add((nny, nnx))

        # remove buildings which are deemed to small. These will not influence the squeezed buildings class, but do contribute to the edges class
        for root, circumference in tree_circumferences.items():
            if circumference < self.min_building_size:
                del first_fronts[root]

        # if there is not two buildings, no need to continue the search
        if len(first_fronts.keys()) < 2:
            return {'gt': full_grid, "path": self.images[idx]}

        all_dists = np.stack([all_dists[key] for key in first_fronts.keys()], axis=-1)
        for r_idx, root in enumerate(first_fronts.keys()):
            front = first_fronts[root]
            visited = visited_by_tree[root]
            closest = closest_from_tree[root]
            for key in visited.keys():
                front.discard(key)
            while len(front) > 0:
                next_front = set()
                for y, x in front:
                    # for the current pixel, we want to add its unvisited neighborus to the next front, and find the closest edge pixels from its visited neigbours
                    visited[(y, x)] = True
                    next_front_candidates = set() # we only update the next front with unvisited neighbours, if we can improve the current closest pixel
                    best_dist = all_dists[y][x][r_idx]
                    best_nodes = closest[(y, x)]
                    for ny, nx in get_all_neighbours(y, x, grid.shape[0], grid.shape[1]):
                        if grid[ny][nx] != 0: # ignore building pixels
                            continue
                        if not visited[(ny, nx)]:
                            if (ny, nx) in front: 
                                # edge case, when a neighbour is not visited but in the current front,
                                # if our closest edge pixel is their closest (currently unknown) edge pixel, this is problematic
                                next_front_candidates.add((y, x))
                                next_front_candidates.add((ny, nx))
                                # therefore we add both the current pixel and neighbour pixel to  the next front and hope for the best
                                # TODO: find out if this works, / is necessary 
                                continue
                            next_front_candidates.add((ny, nx))
                            continue
                        if len(closest[(ny, nx)]) == 0: # the neighbour is visited, but has no edge_pixels which are close enough to pass the max_dist threshhold
                            continue
                        for neighbours_closest_y, ncx in closest[(ny, nx)]:
                            dist = l2(y - neighbours_closest_y, x - ncx)
                            if dist < best_dist:
                                best_dist = dist
                                best_nodes = {(neighbours_closest_y, ncx)}
                            elif dist == best_dist:
                                best_nodes.add((neighbours_closest_y, ncx))
                    if best_dist < self.max_dist:
                        all_dists[y][x][r_idx] = best_dist
                        next_front = next_front.union(next_front_candidates)
                        closest[(y, x)] = best_nodes
                front = next_front

        # find the combined distance of the two closest buildings for each pixel
        if all_dists.shape[-1] > 2:
            shortest_dists_idx = np.argpartition(all_dists, 2)
            best_dists = np.take_along_axis(all_dists, shortest_dists_idx[:, :, :2], axis=-1)
        else:
            best_dists = all_dists
        squeezed_cond = np.sum(best_dists, axis=-1) <= self.max_dist

        # gt_out = grid.copy()
        grid[squeezed_cond] = 3
        grid += classified_building_edges

        # squeezed_background = np.zeros(grid.shape[:2], dtype=np.uint8)
        # squeezed_background[squeezed_cond] = 255
        # classified_building_edges[classified_building_edges > 0] = 255
        # grid[grid > 0] = 255

        # human_gt = np.stack([squeezed_background, classified_building_edges, grid], axis=-1)
        new_gt = np.stack([grid, grid, grid], axis=-1)
        assert new_gt.dtype == np.uint8

        sample = {'gt': new_gt, "path": self.images[idx]} #, "human_gt": human_gt}
        return sample


def main(max_dist, replace_folder="masks", min_building_size=30):
    for split in ["validation"]:
        gt_ds = gtDataset(f"./../../../data/mapai/{split}/masks/*.tif", max_dist, min_building_size)
        dataloader = DataLoader(gt_ds, batch_size=1, shuffle=False, num_workers=8)

        for new_gt in tqdm(dataloader):
            gt = new_gt["gt"].squeeze().numpy()
            path = new_gt["path"][0]

            assert gt.shape == (500, 500, 3)
            assert replace_folder in path

            path = path.replace(replace_folder, replace_folder + "_reclassified")
            cv2.imwrite(path, gt)
            #path = path.replace(replace_folder, replace_folder + "_colored")
            #cv2.imwrite(path[:-4] + ".png", new_gt["human_gt"][0].numpy())

if __name__ == "__main__":
    main(18.0)
