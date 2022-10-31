import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
import re
batch_size = 100

if __name__ == "__main__":
    for type in ["lidar", "masks"]:
        image_paths = [*Path(f"./train/{type}/").glob("*.tif")] + [*Path(f"./validation/{type}").glob("*.tif")]

        image_paths = sorted(image_paths, key = lambda x: [int(k) if k.isdigit() else k for k in re.split('([0-9]+)', x.stem)])

        assert len(image_paths) % batch_size == 0
        left_half = None
        for i in tqdm(range(0, len(image_paths), batch_size)):
            paths = image_paths[i:i+batch_size]
            image_slices = []
            for k in range(0, 100, 10):
                slice = np.hstack([cv2.imread(p.as_posix(), cv2.IMREAD_UNCHANGED) for p in paths[k:k + 10]])
                image_slices.append(slice)

            image = np.vstack(image_slices)
            if left_half is not None:
                image = np.hstack([left_half, image])
                left_half = None
            elif i + batch_size < len(image_paths) and paths[0].stem[:5] == image_paths[i + batch_size].stem[:5] and int(paths[0].stem[5:8]) == int(image_paths[i + batch_size].stem[5:8]) - 1:
                left_half = image
                continue

            new_stem = paths[0].stem[:-2] 
            new_path = Path('big_tiles').joinpath(paths[0]).with_name(new_stem + ".tif")
            new_path.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(new_path.as_posix(), image)
