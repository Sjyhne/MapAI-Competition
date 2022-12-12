import numpy as np


def class_wise(arr: np.array, c: int) -> np.array:
    return arr == c

def iou(prediction, target) -> float:
    prediction, target = prediction.cpu(), target.cpu()
    prediction = prediction.argmax(dim=1)

    miou = []
    for c in range(2):
        pred = class_wise(prediction, c)
        tar = class_wise(target, c)

        if pred.dtype != bool:
            pred = np.asarray(pred, dtype=bool)

        if tar.dtype != bool:
            tar = np.asarray(tar, dtype=bool)

        overlap = pred * tar # Logical AND
        union = pred + tar # Logical OR

        if union.sum() != 0 and overlap.sum() != 0:
            iou = (float(overlap.sum()) / float(union.sum()))
        else:
            iou = 0

        if c in target:
            miou.append(iou)

    return np.asarray(miou).mean()