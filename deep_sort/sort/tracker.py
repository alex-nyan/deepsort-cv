"""
Multi-target tracker: predict → (camera compensate) → associate → update.

This is the main DeepSORT tracking loop with our two modifications:
    1. Kalman filter model selection (CV or CA)
    2. Camera-motion compensation (ECC) inserted between predict and associate

The ablation is controlled by:
    - kalman_model: "constant_velocity" or "constant_acceleration"
    - camera_motion_compensation: True or False
"""

import numpy as np

from .kalman_filter import KalmanFilterCV
from .kalman_filter_accel import KalmanFilterCA
from .track import Track, TrackState
from .nn_matching import NearestNeighborDistanceMetric
from . import linear_assignment
from . import iou_matching


class Tracker:
    """
    Multi-target tracker using DeepSORT's cascade matching.

    Parameters
    ----------
    metric : NearestNeighborDistanceMetric
        The Re-ID distance metric.
    kalman_model : str
        "constant_velocity" (8D) or "constant_acceleration" (12D).
    camera_compensator : ECCCameraCompensator | None
        If provided, applies camera-motion compensation.
    max_iou_distance : float
    max_age : int
    n_init : int
    """

    def __init__(
        self,
        metric,
        kalman_model="constant_velocity",
        camera_compensator=None,
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
        kalman_params=None,
    ):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        # --- Select Kalman filter ---
        kalman_params = kalman_params or {}
        if kalman_model == "constant_velocity":
            self.kf = KalmanFilterCV(
                std_weight_position=kalman_params.get("std_weight_position", 1e-2),
                std_weight_velocity=kalman_params.get("std_weight_velocity", 1e-5),
            )
            self._state_dim = 8
        elif kalman_model == "constant_acceleration":
            self.kf = KalmanFilterCA(
                std_weight_position=kalman_params.get("std_weight_position", 1e-2),
                std_weight_velocity=kalman_params.get("std_weight_velocity", 1e-5),
                std_weight_acceleration=kalman_params.get("std_weight_acceleration", 2.5e-2),
            )
            self._state_dim = 12
        else:
            raise ValueError(f"Unknown kalman_model: {kalman_model}")

        self._kalman_model = kalman_model
        self._camera_compensator = camera_compensator

        self.tracks = []
        self._next_id = 1

    def predict(self, frame=None):
        """
        Propagate all track states to the current frame.

        If camera_compensator is set and frame is provided,
        estimate the camera warp and apply it to all predicted states.

        Parameters
        ----------
        frame : ndarray | None
            Current video frame (needed for ECC compensation).
        """
        # Step 1: Kalman predict
        for track in self.tracks:
            track.predict(self.kf)

        # Step 2: Camera-motion compensation (if enabled)
        if self._camera_compensator is not None and frame is not None:
            warp, success = self._camera_compensator.estimate_warp(frame)
            if success:
                for track in self.tracks:
                    from camera_motion.ecc_compensator import ECCCameraCompensator
                    track.mean = ECCCameraCompensator.warp_state_mean(
                        track.mean, warp, self._state_dim
                    )

    def update(self, detections):
        """
        Perform measurement update and track management.

        Parameters
        ----------
        detections : list of Detection
        """
        # --- Association ---
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update matched tracks
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])

        # Mark unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Initiate new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update Re-ID gallery
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id] * len(track.features)
            track.features = []  # features moved to metric gallery

        self.metric.partial_fit(
            np.asarray(features) if features else np.zeros((0, 0)),
            targets,
            active_targets,
        )

    def _match(self, detections):
        """
        Two-stage matching: cascade (Re-ID) → IoU fallback.
        """

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = [tracks[i].track_id for i in track_indices]

            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets,
                track_indices, detection_indices
            )
            return cost_matrix

        # Split tracks into confirmed and unconfirmed
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Stage 1: Cascade matching on confirmed tracks (Re-ID + gating)
        matches_a, unmatched_tracks_a, unmatched_detections = (
            linear_assignment.matching_cascade(
                gated_metric,
                self.metric.matching_threshold,
                self.max_age,
                self.tracks,
                detections,
                confirmed_tracks,
            )
        )

        # Stage 2: IoU matching on remaining tracks + unconfirmed
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update == 1  # only recently lost
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update != 1
        ]

        matches_b, unmatched_tracks_b, unmatched_detections = (
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost,
                self.max_iou_distance,
                self.tracks,
                detections,
                iou_track_candidates,
                unmatched_detections,
            )
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            feature=detection.feature,
        ))
        self._next_id += 1
