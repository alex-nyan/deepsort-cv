"""
Non-maximum suppression and preprocessing utilities.
"""

import numpy as np


def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    """
    Suppress overlapping detections.

    Parameters
    ----------
    boxes : ndarray (N, 4)
        [top-left-x, top-left-y, width, height]
    max_bbox_overlap : float
        IoU threshold for suppression.
    scores : ndarray (N,) | None
        Detection confidence scores.

    Returns
    -------
    pick : list of int
        Indices of kept detections.
    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float64)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)

    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        union = area[i] + area[idxs[:last]] - intersection
        overlap = intersection / (union + 1e-12)

        idxs = np.delete(idxs, np.concatenate(
            ([last], np.where(overlap > max_bbox_overlap)[0])
        ))

    return pick
