"""
Four-way ablation experiment driver.

Runs all four configurations on all sequences:
    1. baseline:    constant_velocity KF, no camera compensation
    2. accel_only:  constant_acceleration KF, no camera compensation
    3. cmc_only:    constant_velocity KF + ECC camera compensation
    4. full:        constant_acceleration KF + ECC camera compensation

Outputs:
    - Per-sequence MOT metrics (MOTA, IDF1, IDS, FP, FN)
    - Regime analysis (ID switches during accel vs pan events)
    - Comparison tables for the paper

Usage:
    python scripts/run_ablation.py \
        --config configs/defaults.yaml \
        --data_root /path/to/dataset \
        --output_dir results/ablation \
        --reid_model /path/to/reid.pth
"""

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_tracker import run_tracker
from evaluation.mot_metrics import evaluate_sequence, evaluate_dataset
from evaluation.regime_analysis import (
    detect_acceleration_events,
    detect_camera_pan_events,
    count_regime_switches,
    generate_regime_report,
)
from evaluation.mot_metrics import load_mot_results, load_mot_gt


def _load_track_states(filepath):
    """Load a MOTChallenge file into the format expected by regime analysis."""
    from collections import defaultdict
    states = defaultdict(list)
    try:
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                frame = int(parts[0])
                tid = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                cx = x + w / 2
                cy = y + h / 2
                states[frame].append((tid, cx, cy))
    except FileNotFoundError:
        pass
    return dict(states)


def _get_id_switch_frames(gt_file, result_file):
    """
    Approximate ID switch detection: find frames where the tracker's
    assignment to a GT object changes identity.
    """
    try:
        import motmetrics as mm
    except ImportError:
        return []

    gt = load_mot_gt(gt_file)
    results = load_mot_results(result_file)

    import numpy as np
    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(list(gt.keys()) + list(results.keys())))

    for frame in all_frames:
        gt_objs = gt.get(frame, [])
        res_objs = results.get(frame, [])

        gt_ids = [o[0] for o in gt_objs]
        res_ids = [o[0] for o in res_objs]

        gt_boxes = np.array([[o[1], o[2], o[1] + o[3], o[2] + o[4]] for o in gt_objs])
        res_boxes = np.array([[o[1], o[2], o[1] + o[3], o[2] + o[4]] for o in res_objs])

        if len(gt_boxes) > 0 and len(res_boxes) > 0:
            distances = mm.distances.iou_matrix(gt_boxes, res_boxes, max_iou=0.5)
        else:
            distances = np.empty((len(gt_boxes), len(res_boxes)))

        acc.update(gt_ids, res_ids, distances)

    # Extract frames where switches occurred from the events dataframe
    events = acc.events
    switch_events = events[events["Type"] == "SWITCH"]
    switch_frames = []
    for idx in switch_events.index:
        # MOTAccumulator uses (frame_idx, event_idx) as multi-index
        if hasattr(idx, '__iter__'):
            switch_frames.append(all_frames[idx[0]] if idx[0] < len(all_frames) else idx[0])
        else:
            switch_frames.append(idx)

    return switch_frames


ABLATION_CONFIGS = {
    "baseline": {
        "kalman_model": "constant_velocity",
        "camera_motion_compensation": False,
    },
    "accel_only": {
        "kalman_model": "constant_acceleration",
        "camera_motion_compensation": False,
    },
    "cmc_only": {
        "kalman_model": "constant_velocity",
        "camera_motion_compensation": True,
    },
    "full": {
        "kalman_model": "constant_acceleration",
        "camera_motion_compensation": True,
    },
}


def discover_sequences(data_root):
    """Find all sequences in a MOTChallenge-format dataset directory."""
    sequences = []
    for name in sorted(os.listdir(data_root)):
        seq_dir = os.path.join(data_root, name)
        if os.path.isdir(seq_dir) and os.path.exists(os.path.join(seq_dir, "det", "det.txt")):
            sequences.append(name)
    return sequences


def run_ablation(config, data_root, output_dir, reid_model=None):
    """
    Run full four-way ablation.

    Parameters
    ----------
    config : dict
        Base configuration.
    data_root : str
        Dataset root with sequence subdirectories.
    output_dir : str
        Where to write results.
    reid_model : str | None
    """
    sequences = discover_sequences(data_root)
    if not sequences:
        print(f"No sequences found in {data_root}")
        return

    print(f"Found {len(sequences)} sequences: {sequences}")

    all_results = {}

    for config_name, ablation_flags in ABLATION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Running configuration: {config_name}")
        print(f"  Kalman model: {ablation_flags['kalman_model']}")
        print(f"  Camera compensation: {ablation_flags['camera_motion_compensation']}")
        print(f"{'='*60}")

        config_output_dir = os.path.join(output_dir, config_name)
        os.makedirs(config_output_dir, exist_ok=True)

        config_results = {}

        for seq_name in sequences:
            print(f"\n  Sequence: {seq_name}")
            seq_dir = os.path.join(data_root, seq_name)
            output_file = os.path.join(config_output_dir, f"{seq_name}.txt")

            # Run tracker
            warp_magnitudes = run_tracker(
                config,
                seq_dir,
                output_file,
                kalman_model=ablation_flags["kalman_model"],
                camera_motion_compensation=ablation_flags["camera_motion_compensation"],
                reid_model_path=reid_model,
            )

            # Evaluate
            gt_file = os.path.join(seq_dir, "gt", "gt.txt")
            if os.path.exists(gt_file):
                metrics = evaluate_sequence(gt_file, output_file)
                config_results[seq_name] = {
                    "metrics": metrics,
                    "warp_magnitudes": warp_magnitudes or {},
                }
                print(f"    MOTA: {metrics['MOTA']:.4f}  IDF1: {metrics['IDF1']:.4f}  "
                      f"IDS: {metrics['IDS']}")
            else:
                print(f"    No ground truth found, skipping evaluation")

        all_results[config_name] = config_results

    # --- Regime Analysis ---
    print("\n" + "=" * 60)
    print("REGIME ANALYSIS")
    print("=" * 60)

    eval_cfg = config.get("evaluation", {})
    accel_threshold = eval_cfg.get("acceleration_threshold", 15.0)
    pan_threshold = eval_cfg.get("camera_pan_threshold", 5.0)

    regime_counts_by_config = {}

    for config_name, config_results_data in all_results.items():
        total_regime_counts = {"accel_only": 0, "pan_only": 0, "both": 0, "neither": 0, "total": 0}

        for seq_name, data in config_results_data.items():
            if "metrics" not in data:
                continue

            # Build track states from our output for acceleration detection
            output_file = os.path.join(output_dir, config_name, f"{seq_name}.txt")
            if not os.path.exists(output_file):
                continue

            track_states = _load_track_states(output_file)
            gt_file = os.path.join(data_root, seq_name, "gt", "gt.txt")

            # Detect acceleration events from ground truth trajectories
            gt_states = _load_track_states(gt_file)
            accel_frames, _ = detect_acceleration_events(gt_states, threshold=accel_threshold)

            # Detect camera pan events from warp magnitudes
            warp_mags = data.get("warp_magnitudes", {})
            pan_frames, _ = detect_camera_pan_events(warp_mags, threshold=pan_threshold)

            # Get ID switch frames (approximate: frames where our track IDs differ from GT)
            id_switch_frames = _get_id_switch_frames(gt_file, output_file)

            # Count regime breakdown
            regime = count_regime_switches(id_switch_frames, accel_frames, pan_frames)
            for k in total_regime_counts:
                total_regime_counts[k] += regime[k]

        regime_counts_by_config[config_name] = total_regime_counts

    report = generate_regime_report(regime_counts_by_config)
    print(report)

    # --- Generate comparison tables ---
    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)

    # Aggregate metrics table
    header = f"{'Config':<16}{'MOTA':>10}{'IDF1':>10}{'IDS':>10}{'FP':>10}{'FN':>10}"
    print(header)
    print("-" * len(header))

    for config_name in ABLATION_CONFIGS:
        if config_name not in all_results:
            continue
        results = all_results[config_name]
        if not results:
            continue

        total_ids = sum(r["metrics"]["IDS"] for r in results.values() if "metrics" in r)
        total_fp = sum(r["metrics"]["FP"] for r in results.values() if "metrics" in r)
        total_fn = sum(r["metrics"]["FN"] for r in results.values() if "metrics" in r)
        avg_mota = sum(r["metrics"]["MOTA"] for r in results.values() if "metrics" in r) / len(results)
        avg_idf1 = sum(r["metrics"]["IDF1"] for r in results.values() if "metrics" in r) / len(results)

        print(f"{config_name:<16}{avg_mota:>10.4f}{avg_idf1:>10.4f}"
              f"{total_ids:>10}{total_fp:>10}{total_fn:>10}")

    # Save detailed results
    results_file = os.path.join(output_dir, "ablation_results.json")
    serializable = {}
    for config_name, seqs in all_results.items():
        serializable[config_name] = {}
        for seq_name, data in seqs.items():
            serializable[config_name][seq_name] = {
                "metrics": data.get("metrics", {}),
                "regime_analysis": regime_counts_by_config.get(config_name, {}),
            }

    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nDetailed results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Four-way ablation experiment")
    parser.add_argument("--config", default="configs/defaults.yaml")
    parser.add_argument("--data_root", required=True, help="Dataset root directory")
    parser.add_argument("--output_dir", default="results/ablation")
    parser.add_argument("--reid_model", default=None)

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_ablation(config, args.data_root, args.output_dir, args.reid_model)


if __name__ == "__main__":
    main()
