"""
Camera-motion compensation using ECC (Enhanced Correlation Coefficient)
image registration.

In broadcast soccer, the camera pans and zooms continuously. This shifts
every player's apparent position between frames, introducing a systematic
bias into the Kalman filter's predictions. Without compensation, the
predicted bounding box lands in the wrong place, causing association
failures and identity switches.

Approach:
    1. Convert consecutive frames to grayscale.
    2. Optionally downscale and blur for speed and robustness.
    3. Estimate the affine warp matrix W using OpenCV's findTransformECC.
    4. Apply W to the predicted bounding box centers to "undo" the
       camera movement before the association step.

The affine model (6 parameters: rotation, scale, shear, translation)
captures pan, zoom, and mild rotation typical of broadcast cameras.

Integration point:
    Called in Tracker.predict() AFTER the Kalman prediction step but
    BEFORE the association step. The warp is applied to the mean state
    of every active track.
"""

import cv2
import numpy as np


class ECCCameraCompensator:
    """
    Estimates inter-frame camera motion via ECC and applies correction
    to Kalman filter state vectors.
    """

    def __init__(
        self,
        warp_mode="affine",
        num_iterations=50,
        termination_eps=1e-3,
        downscale_factor=2,
        gaussian_blur_sigma=1.5,
    ):
        """
        Parameters
        ----------
        warp_mode : str
            "affine" (6-DOF) or "euclidean" (3-DOF: rotation + translation).
        num_iterations : int
            Maximum ECC iterations.
        termination_eps : float
            ECC convergence threshold.
        downscale_factor : int
            Downscale frames before ECC for speed. 1 = no downscale.
        gaussian_blur_sigma : float
            Gaussian blur sigma applied before ECC for noise robustness.
        """
        warp_modes = {
            "affine": cv2.MOTION_AFFINE,
            "euclidean": cv2.MOTION_EUCLIDEAN,
        }
        if warp_mode not in warp_modes:
            raise ValueError(f"warp_mode must be one of {list(warp_modes)}")

        self._warp_mode = warp_modes[warp_mode]
        self._num_iterations = num_iterations
        self._termination_eps = termination_eps
        self._downscale = downscale_factor
        self._blur_sigma = gaussian_blur_sigma

        self._prev_gray = None
        # Last successfully estimated warp (2x3 affine matrix)
        self._last_warp = np.eye(2, 3, dtype=np.float32)

        self._criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self._num_iterations,
            self._termination_eps,
        )

    def _preprocess(self, frame):
        """Convert to grayscale, downscale, blur."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        if self._downscale > 1:
            h, w = gray.shape
            gray = cv2.resize(
                gray, (w // self._downscale, h // self._downscale)
            )

        if self._blur_sigma > 0:
            ksize = int(2 * round(2 * self._blur_sigma) + 1)
            gray = cv2.GaussianBlur(gray, (ksize, ksize), self._blur_sigma)

        return gray

    def estimate_warp(self, frame):
        """
        Estimate affine warp from the previous frame to the current frame.

        Parameters
        ----------
        frame : ndarray (H, W, 3) or (H, W)
            Current video frame (BGR or grayscale).

        Returns
        -------
        warp_matrix : ndarray (2, 3)
            Affine transformation mapping previous → current frame.
            Identity if this is the first frame or ECC fails.
        success : bool
            Whether ECC converged successfully.
        """
        gray = self._preprocess(frame)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._last_warp = np.eye(2, 3, dtype=np.float32)
            return self._last_warp.copy(), False

        # Initialize warp with identity
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        success = True

        try:
            _, warp_matrix = cv2.findTransformECC(
                self._prev_gray,
                gray,
                warp_matrix,
                self._warp_mode,
                self._criteria,
                inputMask=None,
                gaussFiltSize=1,  # internal ECC blur (minimal, we pre-blur)
            )

            # Scale translation back to full resolution
            if self._downscale > 1:
                warp_matrix[0, 2] *= self._downscale
                warp_matrix[1, 2] *= self._downscale

        except cv2.error:
            # ECC failed to converge — use identity (no compensation)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            success = False

        self._prev_gray = gray
        self._last_warp = warp_matrix
        return warp_matrix, success

    @staticmethod
    def warp_state_mean(mean, warp_matrix, state_dim):
        """
        Apply affine warp to a Kalman filter state vector's position
        (and velocity/acceleration if present).

        The warp transforms the predicted center (cx, cy) from the
        previous frame's coordinate system to the current frame's.

        Parameters
        ----------
        mean : ndarray (state_dim,)
            Kalman state: [cx, cy, a, h, vx, vy, ...].
        warp_matrix : ndarray (2, 3)
            Affine warp (previous → current).
        state_dim : int
            8 for CV model, 12 for CA model.

        Returns
        -------
        warped_mean : ndarray (state_dim,)
        """
        warped = mean.copy()

        # Extract the 2x2 rotation/scale part and 2x1 translation
        R = warp_matrix[:, :2]   # (2, 2)
        t = warp_matrix[:, 2]    # (2,)

        # --- Warp position (cx, cy) ---
        pos = mean[:2]
        warped[:2] = R @ pos + t

        # --- Warp velocity (vx, vy) ---
        # Velocity transforms by R only (no translation)
        if state_dim >= 8:
            vel = mean[4:6]
            warped[4:6] = R @ vel

        # --- Warp acceleration (ax, ay) ---
        # Same linear transform as velocity
        if state_dim >= 12:
            acc = mean[8:10]
            warped[8:10] = R @ acc

        # Aspect ratio and height are approximately invariant under
        # affine camera motion (pan/zoom mostly affects x,y translation
        # and uniform scale, which affects h but not a).
        # We leave a, h, va, vh, aa, ah unchanged.
        # NOTE: for very large zooms, h should also be scaled by det(R).
        # This is a deliberate simplification for broadcast soccer where
        # zoom changes are gradual.

        return warped

    @property
    def last_warp(self):
        """Last estimated warp matrix (2, 3)."""
        return self._last_warp.copy()

    @property
    def last_translation_magnitude(self):
        """Magnitude of last estimated translation (px). Useful for regime analysis."""
        t = self._last_warp[:, 2]
        return float(np.linalg.norm(t))

    def reset(self):
        """Reset state for a new video sequence."""
        self._prev_gray = None
        self._last_warp = np.eye(2, 3, dtype=np.float32)
