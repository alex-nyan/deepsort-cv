"""
Regime analysis: break down identity switches by event type.

The paper proposes a mechanistic breakdown:
    1. Acceleration events: frames where a tracked player's velocity
       changes by more than a threshold (bursty sprints, cuts, stops).
    2. Camera-pan events: frames where the estimated affine warp has
       a translation magnitude above a threshold.

For each regime, we count identity switches that occur within a
temporal window of the event. This tells us *why* each fix helps.
"""

import numpy as np
from collections import defaultdict


def detect_acceleration_events(track_states, threshold=15.0, window=3):
    """
    Identify frames where tracked objects experience high acceleration.

    Parameters
    ----------
    track_states : dict
        frame_id -> list of (track_id, cx, cy)
        Extracted from tracker output at each frame.
    threshold : float
        Minimum velocity change (px/frame) to flag as acceleration event.
    window : int
        Temporal radius: events within ±window frames are grouped.

    Returns
    -------
    accel_frames : set of int
        Frame IDs flagged as acceleration events.
    accel_details : list of dict
        Per-event details for analysis.
    """
    # Build per-track position histories
    histories = defaultdict(list)  # track_id -> [(frame, cx, cy), ...]
    for frame_id, objs in sorted(track_states.items()):
        for tid, cx, cy in objs:
            histories[tid].append((frame_id, cx, cy))

    accel_frames = set()
    accel_details = []

    for tid, history in histories.items():
        if len(history) < 3:
            continue

        history = sorted(history, key=lambda x: x[0])

        for i in range(1, len(history) - 1):
            f_prev, cx_prev, cy_prev = history[i - 1]
            f_curr, cx_curr, cy_curr = history[i]
            f_next, cx_next, cy_next = history[i + 1]

            # Skip non-consecutive frames
            if f_curr - f_prev != 1 or f_next - f_curr != 1:
                continue

            # Velocity at t and t+1
            vx_t = cx_curr - cx_prev
            vy_t = cy_curr - cy_prev
            vx_t1 = cx_next - cx_curr
            vy_t1 = cy_next - cy_curr

            # Acceleration = change in velocity
            ax = vx_t1 - vx_t
            ay = vy_t1 - vy_t
            accel_mag = np.sqrt(ax ** 2 + ay ** 2)

            if accel_mag > threshold:
                for f in range(f_curr - window, f_curr + window + 1):
                    accel_frames.add(f)
                accel_details.append({
                    "frame": f_curr,
                    "track_id": tid,
                    "acceleration": accel_mag,
                })

    return accel_frames, accel_details


def detect_camera_pan_events(warp_magnitudes, threshold=5.0, window=3):
    """
    Identify frames with significant camera motion.

    Parameters
    ----------
    warp_magnitudes : dict
        frame_id -> float (translation magnitude from ECC).
    threshold : float
        Minimum translation (px) to flag as pan event.
    window : int
        Temporal radius.

    Returns
    -------
    pan_frames : set of int
    pan_details : list of dict
    """
    pan_frames = set()
    pan_details = []

    for frame_id, mag in sorted(warp_magnitudes.items()):
        if mag > threshold:
            for f in range(frame_id - window, frame_id + window + 1):
                pan_frames.add(f)
            pan_details.append({
                "frame": frame_id,
                "translation_magnitude": mag,
            })

    return pan_frames, pan_details


def count_regime_switches(id_switch_frames, accel_frames, pan_frames):
    """
    Categorize identity switches by which regime(s) they fall in.

    Parameters
    ----------
    id_switch_frames : list of int
        Frame IDs where identity switches occurred.
    accel_frames : set of int
    pan_frames : set of int

    Returns
    -------
    counts : dict
        "accel_only": switches during acceleration events only
        "pan_only": switches during camera pan events only
        "both": switches during both event types
        "neither": switches outside both regimes
        "total": total identity switches
    """
    counts = {"accel_only": 0, "pan_only": 0, "both": 0, "neither": 0, "total": 0}

    for frame in id_switch_frames:
        in_accel = frame in accel_frames
        in_pan = frame in pan_frames

        if in_accel and in_pan:
            counts["both"] += 1
        elif in_accel:
            counts["accel_only"] += 1
        elif in_pan:
            counts["pan_only"] += 1
        else:
            counts["neither"] += 1

        counts["total"] += 1

    return counts


def generate_regime_report(regime_counts_by_config):
    """
    Generate a formatted comparison table for the paper.

    Parameters
    ----------
    regime_counts_by_config : dict
        config_name -> counts dict from count_regime_switches

    Returns
    -------
    report : str
    """
    configs = list(regime_counts_by_config.keys())
    categories = ["accel_only", "pan_only", "both", "neither", "total"]

    header = f"{'Config':<20}" + "".join(f"{c:>12}" for c in categories)
    lines = [header, "-" * len(header)]

    for config in configs:
        counts = regime_counts_by_config[config]
        row = f"{config:<20}" + "".join(f"{counts[c]:>12}" for c in categories)
        lines.append(row)

    return "\n".join(lines)
