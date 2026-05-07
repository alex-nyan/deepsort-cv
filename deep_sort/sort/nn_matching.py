"""
Nearest-neighbor distance metric for Re-ID feature matching.
"""

import numpy as np


def _pdist(a, b):
    """Compute pairwise squared Euclidean distance."""
    a2 = np.sum(a ** 2, axis=1)
    b2 = np.sum(b ** 2, axis=1)
    return a2[:, None] + b2[None, :] - 2.0 * a @ b.T


def _cosine_distance(a, b):
    """Compute pairwise cosine distance."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return 1.0 - a @ b.T


class NearestNeighborDistanceMetric:
    """
    Nearest-neighbor distance metric for Re-ID.

    For each track, maintains a gallery of recent appearance features.
    Distance from a detection to a track is the minimum distance
    between the detection feature and any feature in the track's gallery.

    Parameters
    ----------
    metric : str
        "cosine" or "euclidean".
    matching_threshold : float
        Gating threshold for the metric.
    budget : int | None
        Maximum gallery size per track. Oldest features are evicted.
    """

    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "cosine":
            self._metric = _cosine_distance
        elif metric == "euclidean":
            self._metric = lambda a, b: _pdist(a, b)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}  # track_id -> list of features

    def partial_fit(self, features, targets, active_targets):
        """
        Update the gallery with new features.

        Parameters
        ----------
        features : ndarray (N, D)
        targets : list of int
            Track IDs corresponding to each feature.
        active_targets : list of int
            Currently active track IDs (prune others).
        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget :]

        # Remove inactive tracks
        self.samples = {k: v for k, v in self.samples.items() if k in active_targets}

    def distance(self, features, targets):
        """
        Compute cost matrix between detection features and track galleries.

        Parameters
        ----------
        features : ndarray (N, D)
            Detection features.
        targets : list of int
            Track IDs (one per row of cost matrix).

        Returns
        -------
        cost_matrix : ndarray (len(targets), N)
        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            gallery = np.array(self.samples.get(target, []))
            if len(gallery) == 0:
                cost_matrix[i, :] = self.matching_threshold + 1e-5
            else:
                distances = self._metric(gallery, features)
                cost_matrix[i, :] = distances.min(axis=0)
        return cost_matrix
