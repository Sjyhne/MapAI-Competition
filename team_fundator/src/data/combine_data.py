import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
import re
batch_size = 10**2 # images are either 10 * 10 or (10 * 2) * 10

if __name__ == "__main__":
    for type in ["images", "lidar", "masks"]:
        image_paths = [*Path(f"./../../../data/train/{type}/").glob("*.tif")] + [*Path(f"./../../../data/validation/{type}").glob("*.tif")]

        image_paths = sorted(image_paths, key = lambda x: [int(k) if k.isdigit() else k for k in re.split('([0-9]+)', x.stem)])

        assert len(image_paths) % batch_size == 0
        left_half = None # used when combining two adjacent 10 * 10 image tiles
        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Stitching {type}"):
            paths = image_paths[i:i+batch_size]
            image_slices = []
            for k in range(0, 100, 10):
                # collect horisontal slices
                slice = np.hstack([cv2.imread(p.as_posix(), cv2.IMREAD_UNCHANGED) for p in paths[k:k + 10]])
                image_slices.append(slice)

            # stack horisontal image slices
            image = np.vstack(image_slices)
            if left_half is not None:
                # if left half is not None, the current image is adjacent and can be stacked into a 10000 x 5000 image
                image = np.hstack([left_half, image])
                left_half = None
            # check filename of the next image tile to see if it is adjacent to image
            elif i + batch_size < len(image_paths) and paths[0].stem[:5] == image_paths[i + batch_size].stem[:5] and int(paths[0].stem[5:8]) == int(image_paths[i + batch_size].stem[5:8]) - 1:
                left_half = image
                continue
            
            # save the big tile
            new_stem = paths[0].stem[:-2] 
            new_path = Path('./../../../data/big_tiles/')
            new_path = new_path.joinpath(paths[0].relative_to(paths[0].parents[2])).with_name(new_stem + ".tif")

            new_path.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(new_path.as_posix(), image)
