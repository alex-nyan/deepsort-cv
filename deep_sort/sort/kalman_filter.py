"""
Standard DeepSORT Kalman filter: 8-dimensional constant-velocity model.

State vector (8D):
    [cx, cy, a, h, vx, vy, va, vh]

    cx, cy  — bounding box center
    a       — aspect ratio (width / height)
    h       — bounding box height
    vx, vy, va, vh — respective velocities

Measurement vector (4D):
    [cx, cy, a, h]

Motion model:
    x_{t+1} = F @ x_t + noise
    where F is the constant-velocity transition matrix.
"""

import numpy as np
import scipy.linalg


class KalmanFilterCV:
    """
    Constant-velocity Kalman filter for bounding box tracking.
    This is the standard DeepSORT motion model.
    """

    ndim = 4  # measurement dimensions
    dt = 1.0  # frame interval

    def __init__(self, std_weight_position=1e-2, std_weight_velocity=1e-5):
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

        # State transition matrix F (8x8): constant velocity
        self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt

        # Measurement matrix H (4x8): observe position only
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)

    def initiate(self, measurement):
        """
        Create track from initial bounding box measurement.

        Parameters
        ----------
        measurement : ndarray (4,)
            [cx, cy, a, h]

        Returns
        -------
        mean : ndarray (8,)
        covariance : ndarray (8, 8)
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # Initial covariance: large uncertainty in velocity
        std = [
            2 * self._std_weight_position * measurement[3],   # cx
            2 * self._std_weight_position * measurement[3],   # cy
            1e-2,                                              # a
            2 * self._std_weight_position * measurement[3],   # h
            10 * self._std_weight_velocity * measurement[3],  # vx
            10 * self._std_weight_velocity * measurement[3],  # vy
            1e-5,                                              # va
            10 * self._std_weight_velocity * measurement[3],  # vh
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Run Kalman prediction step.

        Parameters
        ----------
        mean : ndarray (8,)
        covariance : ndarray (8, 8)

        Returns
        -------
        mean : ndarray (8,)
        covariance : ndarray (8, 8)
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self._motion_mat @ mean
        covariance = (
            self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        )
        return mean, covariance

    def project(self, mean, covariance):
        """
        Project state to measurement space.

        Returns
        -------
        projected_mean : ndarray (4,)
        projected_covariance : ndarray (4, 4)
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = self._update_mat @ mean
        covariance = (
            self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        )
        return mean, covariance

    def update(self, mean, covariance, measurement):
        """
        Run Kalman correction step.

        Parameters
        ----------
        mean : ndarray (8,)
        covariance : ndarray (8, 8)
        measurement : ndarray (4,)

        Returns
        -------
        new_mean : ndarray (8,)
        new_covariance : ndarray (8, 8)
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # Kalman gain
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            (covariance @ self._update_mat.T).T,
            check_finite=False,
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """
        Compute Mahalanobis gating distance.

        Parameters
        ----------
        mean : ndarray (8,)
        covariance : ndarray (8, 8)
        measurements : ndarray (N, 4)
        only_position : bool
            If True, use only (cx, cy) for gating.

        Returns
        -------
        squared_maha : ndarray (N,)
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - projected_mean
        cholesky = np.linalg.cholesky(projected_cov)
        z = scipy.linalg.solve_triangular(
            cholesky, d.T, lower=True, check_finite=False
        )
        return np.sum(z * z, axis=0)
