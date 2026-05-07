"""
Linear assignment with cascade matching.

DeepSORT's cascade matching gives priority to tracks that were recently
seen (lower time_since_update), which helps preserve identity through
short occlusions.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

INFTY_COST = 1e5


def min_cost_matching(distance_metric, max_distance, tracks, detections,
                      track_indices=None, detection_indices=None):
    """
    Solve the linear assignment problem.

    Parameters
    ----------
    distance_metric : callable
        Signature: (tracks, detections, track_indices, detection_indices) -> cost_matrix
    max_distance : float
        Entries above this are set to INFTY_COST.
    tracks : list of Track
    detections : list of Detection
    track_indices : list of int | None
    detection_indices : list of int | None

    Returns
    -------
    matches : list of (track_idx, detection_idx)
    unmatched_tracks : list of int
    unmatched_detections : list of int
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)

    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(distance_metric, max_distance, cascade_depth, tracks,
                     detections, track_indices=None, detection_indices=None):
    """
    Run matching cascade: match tracks by increasing time_since_update.

    This gives priority to recently-seen tracks, reducing the chance
    that a detection is stolen by a long-lost track.

    Parameters
    ----------
    distance_metric : callable
    max_distance : float
    cascade_depth : int
        Number of cascade levels (= tracker.max_age).
    tracks : list of Track
    detections : list of Detection
    track_indices : list of int | None
        Tracks to consider (typically confirmed tracks only).
    detection_indices : list of int | None

    Returns
    -------
    matches, unmatched_tracks, unmatched_detections
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = list(detection_indices)
    matches = []

    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break

        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        if len(track_indices_l) == 0:
            continue

        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric, max_distance, tracks, detections,
            track_indices_l, unmatched_detections
        )
        matches += matches_l

    unmatched_tracks = [
        k for k in track_indices
        if k not in {m[0] for m in matches}
    ]
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices,
                     detection_indices, gated_cost=INFTY_COST,
                     only_position=False):
    """
    Apply Mahalanobis gating to a cost matrix.

    Invalidate entries where the detection is far from the track's
    predicted position (unlikely to be the same target).
    """
    # Chi-squared 95% threshold for 4 DOF
    gating_threshold = 9.4877

    measurements = np.asarray([
        detections[i].to_xyah() for i in detection_indices
    ])

    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost

    return cost_matrix
