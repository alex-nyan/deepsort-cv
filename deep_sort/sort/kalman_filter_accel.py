"""
Constant-acceleration Kalman filter: 12-dimensional state model.

State vector (12D):
    [cx, cy, a, h, vx, vy, va, vh, ax, ay, aa, ah]

    cx, cy  — bounding box center
    a       — aspect ratio (width / height)
    h       — bounding box height
    vx, vy, va, vh — respective velocities
    ax, ay, aa, ah — respective accelerations

Measurement vector (4D):
    [cx, cy, a, h]

Motion model (constant acceleration kinematics):
    p_{t+1} = p_t + v_t * dt + 0.5 * a_t * dt^2
    v_{t+1} = v_t + a_t * dt
    a_{t+1} = a_t   (constant acceleration assumption)

This captures the bursty, nonlinear motion of soccer players:
sprints, sharp cuts, and sudden stops produce large accelerations
that the standard constant-velocity model cannot track.

Design decisions:
    - Acceleration noise is deliberately high (std_weight_acceleration)
      so the filter can absorb sudden direction changes without
      becoming overconfident in a stale acceleration estimate.
    - Position/velocity noise parameters match the CV baseline
      for fair ablation comparison.
"""

import numpy as np
import scipy.linalg


class KalmanFilterCA:
    """
    Constant-acceleration Kalman filter for bounding box tracking.
    12D state = [position(4), velocity(4), acceleration(4)].
    """

    ndim = 4   # measurement dimensions (cx, cy, a, h)
    dt = 1.0   # frame interval

    def __init__(
        self,
        std_weight_position=1e-2,
        std_weight_velocity=1e-5,
        std_weight_acceleration=2.5e-2,
    ):
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity
        self._std_weight_acceleration = std_weight_acceleration

        # --- State transition matrix F (12x12) ---
        # Kinematic model:  p' = p + v*dt + 0.5*a*dt^2
        #                   v' = v + a*dt
        #                   a' = a
        self._motion_mat = np.eye(3 * self.ndim)
        dt = self.dt
        for i in range(self.ndim):
            # position ← velocity
            self._motion_mat[i, self.ndim + i] = dt
            # position ← acceleration (0.5 * dt^2)
            self._motion_mat[i, 2 * self.ndim + i] = 0.5 * dt * dt
            # velocity ← acceleration
            self._motion_mat[self.ndim + i, 2 * self.ndim + i] = dt

        # --- Measurement matrix H (4x12) ---
        # We observe position only: [cx, cy, a, h]
        self._update_mat = np.eye(self.ndim, 3 * self.ndim)

    def initiate(self, measurement):
        """
        Create track from initial bounding box measurement.

        Parameters
        ----------
        measurement : ndarray (4,)
            [cx, cy, a, h]

        Returns
        -------
        mean : ndarray (12,)
        covariance : ndarray (12, 12)
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean_acc = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel, mean_acc]

        h = measurement[3]  # height as scale reference

        std = [
            # Position uncertainty
            2 * self._std_weight_position * h,    # cx
            2 * self._std_weight_position * h,    # cy
            1e-2,                                  # a
            2 * self._std_weight_position * h,    # h
            # Velocity uncertainty (high — we have no velocity info yet)
            10 * self._std_weight_velocity * h,   # vx
            10 * self._std_weight_velocity * h,   # vy
            1e-5,                                  # va
            10 * self._std_weight_velocity * h,   # vh
            # Acceleration uncertainty (very high — unknown)
            10 * self._std_weight_acceleration * h,  # ax
            10 * self._std_weight_acceleration * h,  # ay
            1e-3,                                     # aa
            10 * self._std_weight_acceleration * h,  # ah
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Run Kalman prediction step using constant-acceleration model.

        Parameters
        ----------
        mean : ndarray (12,)
        covariance : ndarray (12, 12)

        Returns
        -------
        mean : ndarray (12,)
        covariance : ndarray (12, 12)
        """
        h = mean[3]  # current height estimate for noise scaling

        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        std_acc = [
            self._std_weight_acceleration * h,
            self._std_weight_acceleration * h,
            1e-3,
            self._std_weight_acceleration * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel, std_acc]))

        mean = self._motion_mat @ mean
        covariance = (
            self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        )
        return mean, covariance

    def project(self, mean, covariance):
        """
        Project state distribution to measurement space.

        Returns
        -------
        projected_mean : ndarray (4,)
        projected_cov : ndarray (4, 4)
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        projected_mean = self._update_mat @ mean
        projected_cov = (
            self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        )
        return projected_mean, projected_cov

    def update(self, mean, covariance, measurement):
        """
        Run Kalman correction step.

        Parameters
        ----------
        mean : ndarray (12,)
        covariance : ndarray (12, 12)
        measurement : ndarray (4,)

        Returns
        -------
        new_mean : ndarray (12,)
        new_covariance : ndarray (12, 12)
        """
        projected_mean, projected_cov = self.project(mean, covariance)

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
        Compute squared Mahalanobis distance for gating.

        Parameters
        ----------
        mean : ndarray (12,)
        covariance : ndarray (12, 12)
        measurements : ndarray (N, 4)
        only_position : bool
            If True, gate using only (cx, cy).

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

    # --- Convenience accessors for the tracker ---

    def state_to_bbox(self, mean):
        """Extract [cx, cy, a, h] from state vector."""
        return mean[:self.ndim].copy()

    def state_velocity(self, mean):
        """Extract [vx, vy, va, vh] from state vector."""
        return mean[self.ndim : 2 * self.ndim].copy()

    def state_acceleration(self, mean):
        """Extract [ax, ay, aa, ah] from state vector."""
        return mean[2 * self.ndim :].copy()
