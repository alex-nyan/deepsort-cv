"""
Single-target track with state machine (tentative → confirmed → deleted).

Works with both the 8D constant-velocity and 12D constant-acceleration
Kalman filters — the track doesn't care about the state dimension,
it delegates all motion logic to the KF instance.
"""

import numpy as np


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single tracked target.

    Parameters
    ----------
    mean : ndarray
        Initial Kalman state mean.
    covariance : ndarray
        Initial Kalman state covariance.
    track_id : int
        Unique track identifier.
    n_init : int
        Hits needed before confirmation.
    max_age : int
        Frames before deletion of unmatched track.
    feature : ndarray | None
        Initial Re-ID feature.
    """

    _count = 0

    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """Get current position as (top-left-x, top-left-y, width, height)."""
        # State: [cx, cy, a, h, ...]
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]         # w = a * h
        ret[:2] -= ret[2:] / 2   # top-left = center - size/2
        return ret

    def to_tlbr(self):
        """Get current position as (x1, y1, x2, y2)."""
        ret = self.to_tlwh()
        ret[2:] += ret[:2]
        return ret

    def predict(self, kf):
        """
        Propagate state distribution to current frame.

        Parameters
        ----------
        kf : KalmanFilterCV or KalmanFilterCA
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """
        Perform Kalman measurement update and update feature gallery.

        Parameters
        ----------
        kf : KalmanFilterCV or KalmanFilterCA
        detection : Detection
        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah()
        )
        if detection.feature is not None:
            self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0

        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association this frame)."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted
