"""
Evaluate Kalman filter prediction accuracy against true future frames.

For each tracked object, we:
  1. Feed ground-truth positions as measurements up to frame T.
  2. Predict forward K frames (K=1,5,10,15,20) using only the motion model.
  3. Compare predicted center (cx, cy) against true center at frame T+K.
  4. Report mean/median displacement error in pixels.

This directly answers: "How well does the motion model predict where objects go next?"

We compare both Kalman filter variants:
  - Constant-velocity (8D) — original DeepSORT
  - Constant-acceleration (12D) — our extension
"""

import os
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deep_sort.sort.kalman_filter import KalmanFilterCV
from deep_sort.sort.kalman_filter_accel import KalmanFilterCA


def load_ground_truth(gt_file):
    """
    Load ground truth in MOTChallenge format.

    Returns
    -------
    tracks : dict
        track_id -> sorted list of (frame, cx, cy, a, h)
    """
    tracks = defaultdict(list)
    with open(gt_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            cx = x + w / 2
            cy = y + h / 2
            a = w / h if h > 0 else 0
            tracks[track_id].append((frame, cx, cy, a, h))

    # Sort each track by frame
    for tid in tracks:
        tracks[tid].sort(key=lambda x: x[0])
    return tracks


def tlwh_to_xyah(x, y, w, h):
    """Convert top-left-width-height to center-x, center-y, aspect, height."""
    cx = x + w / 2
    cy = y + h / 2
    a = w / h if h > 0 else 0
    return np.array([cx, cy, a, h])


def evaluate_prediction(gt_file, prediction_horizons=(1, 5, 10, 15, 20),
                        min_history=10, eval_interval=10):
    """
    Evaluate multi-step prediction accuracy for both KF models.

    Parameters
    ----------
    gt_file : str
        Path to ground truth file.
    prediction_horizons : tuple of int
        How many frames ahead to predict.
    min_history : int
        Minimum frames of measurements before testing prediction.
    eval_interval : int
        Evaluate prediction every N frames per track.
    """
    tracks = load_ground_truth(gt_file)
    max_horizon = max(prediction_horizons)

    # Results: model_name -> horizon -> list of displacement errors
    results = {
        "constant_velocity": defaultdict(list),
        "constant_acceleration": defaultdict(list),
    }

    kf_cv = KalmanFilterCV()
    kf_ca = KalmanFilterCA()

    num_tracks = len(tracks)
    num_evaluations = 0

    for track_id, observations in tracks.items():
        if len(observations) < min_history + max_horizon:
            continue

        frames = [obs[0] for obs in observations]

        # Test prediction at regular intervals along the track
        for eval_start_idx in range(min_history, len(observations) - max_horizon, eval_interval):
            eval_frame = frames[eval_start_idx]

            # --- Constant Velocity (8D) ---
            # Initialize and feed history
            first_obs = observations[0]
            measurement_cv = np.array(first_obs[1:])  # cx, cy, a, h
            mean_cv, cov_cv = kf_cv.initiate(measurement_cv)

            for i in range(1, eval_start_idx + 1):
                mean_cv, cov_cv = kf_cv.predict(mean_cv, cov_cv)
                meas = np.array(observations[i][1:])
                mean_cv, cov_cv = kf_cv.update(mean_cv, cov_cv, meas)

            # Now predict forward without measurements
            pred_mean_cv = mean_cv.copy()
            pred_cov_cv = cov_cv.copy()
            for h in range(1, max_horizon + 1):
                pred_mean_cv, pred_cov_cv = kf_cv.predict(pred_mean_cv, pred_cov_cv)

                if h in prediction_horizons:
                    future_idx = eval_start_idx + h
                    if future_idx < len(observations):
                        true_cx, true_cy = observations[future_idx][1], observations[future_idx][2]
                        pred_cx, pred_cy = pred_mean_cv[0], pred_mean_cv[1]
                        error = np.sqrt((pred_cx - true_cx)**2 + (pred_cy - true_cy)**2)
                        results["constant_velocity"][h].append(error)

            # --- Constant Acceleration (12D) ---
            measurement_ca = np.array(first_obs[1:])
            mean_ca, cov_ca = kf_ca.initiate(measurement_ca)

            for i in range(1, eval_start_idx + 1):
                mean_ca, cov_ca = kf_ca.predict(mean_ca, cov_ca)
                meas = np.array(observations[i][1:])
                mean_ca, cov_ca = kf_ca.update(mean_ca, cov_ca, meas)

            # Predict forward without measurements
            pred_mean_ca = mean_ca.copy()
            pred_cov_ca = cov_ca.copy()
            for h in range(1, max_horizon + 1):
                pred_mean_ca, pred_cov_ca = kf_ca.predict(pred_mean_ca, pred_cov_ca)

                if h in prediction_horizons:
                    future_idx = eval_start_idx + h
                    if future_idx < len(observations):
                        true_cx, true_cy = observations[future_idx][1], observations[future_idx][2]
                        pred_cx, pred_cy = pred_mean_ca[0], pred_mean_ca[1]
                        error = np.sqrt((pred_cx - true_cx)**2 + (pred_cy - true_cy)**2)
                        results["constant_acceleration"][h].append(error)

            num_evaluations += 1

    return results, num_evaluations, num_tracks


def print_results(results, num_evaluations, num_tracks):
    """Print formatted comparison table."""
    print("\n" + "=" * 70)
    print("PREDICTION ACCURACY EVALUATION")
    print("=" * 70)
    print(f"Tracks evaluated: {num_tracks}")
    print(f"Total prediction trials: {num_evaluations}")
    print()

    horizons = sorted(results["constant_velocity"].keys())

    # Header
    print(f"{'Horizon':>10} | {'CV (8D) Mean':>14} {'CV Median':>12} | "
          f"{'CA (12D) Mean':>14} {'CA Median':>12} | {'Improvement':>12}")
    print("-" * 85)

    for h in horizons:
        cv_errors = results["constant_velocity"][h]
        ca_errors = results["constant_acceleration"][h]

        if not cv_errors or not ca_errors:
            continue

        cv_mean = np.mean(cv_errors)
        cv_median = np.median(cv_errors)
        ca_mean = np.mean(ca_errors)
        ca_median = np.median(ca_errors)

        improvement = (cv_mean - ca_mean) / cv_mean * 100

        print(f"{h:>7} fr | {cv_mean:>10.2f} px {cv_median:>10.2f} px | "
              f"{ca_mean:>10.2f} px {ca_median:>10.2f} px | "
              f"{improvement:>+9.1f}%")

    print()

    # Per-horizon breakdown
    print("Displacement Error Distribution (pixels):")
    print("-" * 70)
    print(f"{'Horizon':>10} | {'Model':>16} | {'Mean':>8} {'Std':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'P95':>8}")
    print("-" * 70)

    for h in horizons:
        for model_name in ["constant_velocity", "constant_acceleration"]:
            errors = results[model_name][h]
            if not errors:
                continue
            arr = np.array(errors)
            label = "CV (8D)" if "velocity" in model_name else "CA (12D)"
            print(f"{h:>7} fr | {label:>16} | {arr.mean():>8.2f} {arr.std():>8.2f} "
                  f"{np.percentile(arr, 25):>8.2f} {np.percentile(arr, 50):>8.2f} "
                  f"{np.percentile(arr, 75):>8.2f} {np.percentile(arr, 95):>8.2f}")
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate KF prediction accuracy")
    parser.add_argument("--gt_file", required=True, help="Ground truth file (MOTChallenge format)")
    parser.add_argument("--min_history", type=int, default=10,
                        help="Min frames of observation before predicting")
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="Evaluate prediction every N frames")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 15, 20],
                        help="Prediction horizons (frames)")
    args = parser.parse_args()

    results, num_eval, num_tracks = evaluate_prediction(
        args.gt_file,
        prediction_horizons=tuple(args.horizons),
        min_history=args.min_history,
        eval_interval=args.eval_interval,
    )

    print_results(results, num_eval, num_tracks)


if __name__ == "__main__":
    main()
