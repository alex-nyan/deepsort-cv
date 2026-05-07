"""
IoU-based association as fallback for unmatched tracks/detections.
"""

import numpy as np
from . import linear_assignment


def iou(bbox, candidates):
    """
    Compute IoU between a bounding box and candidate bounding boxes.

    Parameters
    ----------
    bbox : ndarray (4,)
        [top-left-x, top-left-y, bottom-right-x, bottom-right-y]
    candidates : ndarray (N, 4)
        Same format.

    Returns
    -------
    iou : ndarray (N,)
    """
    bbox_tl = bbox[:2]
    bbox_br = bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, 2:]

    tl = np.maximum(bbox_tl, candidates_tl)
    br = np.minimum(bbox_br, candidates_br)
    wh = np.maximum(0.0, br - tl)

    area_intersection = wh[:, 0] * wh[:, 1]
    area_bbox = np.prod(bbox_br - bbox_tl)
    area_candidates = np.prod(candidates_br - candidates_tl, axis=1)

    return area_intersection / (area_bbox + area_candidates - area_intersection + 1e-12)


def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """
    Build IoU-based cost matrix.

    Parameters
    ----------
    tracks : list of Track
    detections : list of Detection
    track_indices, detection_indices : list of int | None

    Returns
    -------
    cost_matrix : ndarray (len(track_indices), len(detection_indices))
        1 - IoU, so lower is better.
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        # Convert track state to tlbr
        bbox = tracks[track_idx].to_tlbr()
        candidates = np.array([detections[i].to_tlbr() for i in detection_indices])
        if len(candidates) == 0:
            continue
        cost_matrix[row, :] = 1.0 - iou(bbox, candidates)

    return cost_matrix
