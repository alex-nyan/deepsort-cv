"""
Detection data class for a single bounding box detection.
"""

import numpy as np


class Detection:
    """
    A single bounding box detection in a frame.

    Parameters
    ----------
    tlwh : array_like
        Bounding box [top-left-x, top-left-y, width, height].
    confidence : float
        Detection confidence score.
    feature : array_like | None
        Re-ID appearance feature vector.
    """

    def __init__(self, tlwh, confidence, feature=None):
        self.tlwh = np.asarray(tlwh, dtype=np.float64)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32) if feature is not None else None

    def to_tlbr(self):
        """Convert to [top-left-x, top-left-y, bottom-right-x, bottom-right-y]."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert to [center-x, center-y, aspect-ratio, height]."""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2  # center
        ret[2] /= ret[3]        # aspect ratio = w/h
        return ret
