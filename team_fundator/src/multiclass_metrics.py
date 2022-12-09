import numpy as np
import cv2

def iou(prediction: np.array, target: np.array) -> float:

    if prediction.dtype != bool:
        prediction = np.asarray(prediction, dtype=bool)

    if target.dtype != bool:
        target = np.asarray(target, dtype=bool)

    overlap = prediction * target # Logical AND
    union = prediction + target # Logical OR

    if union.sum() != 0 and overlap.sum() != 0:
        iou = (float(overlap.sum()) / float(union.sum()))
    else:
        iou = 0

    return iou

# General util function to get the boundary of a binary mask.
def _mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def biou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    gt_boundary = _mask_to_boundary(gt, dilation_ratio)
    dt_boundary = _mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    if union == 0 or intersection == 0:
        boundary_iou = 0
    else:
        boundary_iou = (intersection / union)

    return boundary_iou

def calculate_multiclass_score(preds: np.array, tars: np.array, num_classes=None) -> dict:

    assert preds.shape == tars.shape, f"pred shape {preds.shape} does not match tar shape {tars.shape}"
    assert len(preds.shape) != 4, f"expected shape is (bs, ydim, xdim), but found {preds.shape}"
    assert type(preds) == np.ndarray, f"preds is a {type(preds)}, but should be numpy.ndarray"
    assert type(tars) == np.ndarray, f"tars is a {type(tars)}, but should be numpy.ndarray"
    assert type(preds[0][0][0]) == np.uint8, f"preds is not of type np.uint8, but {type(preds[0][0][0])}"
    assert type(tars[0][0][0]) == np.uint8, f"tars is not of type np.uint8, but {type(tars[0][0][0])}"

    bs = preds.shape[0]

    t_ious = np.zeros(bs)
    t_bious = np.zeros(bs)

    if num_classes is None:
        num_classes = max(np.max(preds), np.max(tars))
    if num_classes <= 1:
        num_classes = 2

    for i in range(bs):
        class_count = 0
        for class_idx in range(num_classes):
            target = tars[i] == class_idx
            if np.any(target):
                pred = preds[i] == class_idx
                t_ious[i] += iou(pred, target)
                t_bious[i] += biou(pred.astype(np.uint8), target.astype(np.uint8))
                class_count += 1
        t_ious[i] /= class_count
        t_bious[i] /= class_count

    t_iou = np.mean(t_ious)
    t_biou = np.mean(t_bious)

    score = (t_iou + t_biou) / 2
    return {"score": score, "iou": t_iou, "biou": t_biou}