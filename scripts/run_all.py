"""
Run the complete evaluation pipeline on SportsMOT soccer data.

This is a single entry point that:
    1. Generates YOLOv8 detections for sequences missing det/det.txt
    2. Runs the four-way DeepSORT ablation (baseline, accel_only, cmc_only, full)
    3. Runs KF prediction accuracy evaluation (CV 8D vs CA 12D)
    4. Aggregates results across train+val splits and prints paper-ready tables

Usage:
    # Run everything (will take a while on first run due to detection generation):
    python scripts/run_all.py

    # Skip detection generation if you already have det.txt files:
    python scripts/run_all.py --skip_detections

    # Skip individual evaluation stages:
    python scripts/run_all.py --skip_prediction

    # Use a specific YOLO model:
    python scripts/run_all.py --yolo_model yolov8x.pt
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "defaults.yaml")
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "soccer")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "full_eval")


# ── Step 1: Detection generation ─────────────────────────────────────────────

def generate_missing_detections(data_root, splits, yolo_model="yolov8x.pt",
                                conf=0.3, imgsz=1280):
    """Generate YOLOv8 detections for any sequence missing det/det.txt."""
    from ultralytics import YOLO

    missing = []
    for split in splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        for seq in sorted(os.listdir(split_dir)):
            seq_dir = os.path.join(split_dir, seq)
            det_file = os.path.join(seq_dir, "det", "det.txt")
            if os.path.isdir(os.path.join(seq_dir, "img1")) and not os.path.exists(det_file):
                missing.append((split, seq, seq_dir))

    if not missing:
        print("[Step 1] All sequences already have detections. Skipping.\n")
        return

    print(f"[Step 1] Generating detections for {len(missing)} sequences "
          f"using {yolo_model}...")
    model = YOLO(yolo_model)

    from tools.generate_detections import generate_dets_for_sequence

    for i, (split, seq, seq_dir) in enumerate(missing, 1):
        print(f"  [{i}/{len(missing)}] {split}/{seq}")
        generate_dets_for_sequence(model, seq_dir, conf_threshold=conf, imgsz=imgsz)

    print(f"[Step 1] Detection generation complete.\n")


# ── Step 2: Four-way ablation ────────────────────────────────────────────────

def run_ablation_all_splits(config, data_root, splits, output_dir, reid_model=None):
    """Run four-way ablation across multiple splits, aggregate results."""
    from scripts.run_tracker import run_tracker
    from evaluation.mot_metrics import evaluate_sequence
    from evaluation.regime_analysis import (
        detect_acceleration_events,
        detect_camera_pan_events,
        count_regime_switches,
        generate_regime_report,
    )

    CONFIGS = {
        "baseline":   {"kalman_model": "constant_velocity",     "camera_motion_compensation": False},
        "accel_only": {"kalman_model": "constant_acceleration", "camera_motion_compensation": False},
        "cmc_only":   {"kalman_model": "constant_velocity",     "camera_motion_compensation": True},
        "full":       {"kalman_model": "constant_acceleration", "camera_motion_compensation": True},
    }

    sequences = []
    for split in splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        for seq in sorted(os.listdir(split_dir)):
            seq_dir = os.path.join(split_dir, seq)
            det_file = os.path.join(seq_dir, "det", "det.txt")
            gt_file = os.path.join(seq_dir, "gt", "gt.txt")
            if os.path.exists(det_file) and os.path.exists(gt_file):
                sequences.append((split, seq, seq_dir))

    if not sequences:
        print("[Step 2] No sequences with both det.txt and gt.txt found. Skipping.\n")
        return {}

    print(f"[Step 2] Running four-way ablation on {len(sequences)} sequences...")

    all_results = {}

    for config_name, flags in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Config: {config_name}")
        print(f"  Kalman: {flags['kalman_model']}  |  CMC: {flags['camera_motion_compensation']}")
        print(f"{'='*60}")

        config_dir = os.path.join(output_dir, "ablation", config_name)
        os.makedirs(config_dir, exist_ok=True)

        config_results = {}

        for split, seq, seq_dir in sequences:
            tag = f"{split}_{seq}"
            output_file = os.path.join(config_dir, f"{tag}.txt")
            print(f"  [{tag}] ", end="", flush=True)

            t0 = time.time()
            warp_magnitudes = run_tracker(
                config, seq_dir, output_file,
                kalman_model=flags["kalman_model"],
                camera_motion_compensation=flags["camera_motion_compensation"],
                reid_model_path=reid_model,
            )

            gt_file = os.path.join(seq_dir, "gt", "gt.txt")
            metrics = evaluate_sequence(gt_file, output_file)
            elapsed = time.time() - t0

            config_results[tag] = {
                "metrics": metrics,
                "warp_magnitudes": warp_magnitudes or {},
            }
            print(f"MOTA={metrics['MOTA']:.4f}  IDF1={metrics['IDF1']:.4f}  "
                  f"IDS={metrics['IDS']}  ({elapsed:.1f}s)")

        all_results[config_name] = config_results

    # ── Regime analysis ──
    print(f"\n{'='*60}")
    print("  REGIME ANALYSIS")
    print(f"{'='*60}")

    eval_cfg = config.get("evaluation", {})
    accel_thresh = eval_cfg.get("acceleration_threshold", 15.0)
    pan_thresh = eval_cfg.get("camera_pan_threshold", 5.0)

    def _load_track_states(filepath):
        states = defaultdict(list)
        try:
            with open(filepath) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 6:
                        continue
                    frame = int(parts[0])
                    tid = int(parts[1])
                    cx = float(parts[2]) + float(parts[4]) / 2
                    cy = float(parts[3]) + float(parts[5]) / 2
                    states[frame].append((tid, cx, cy))
        except FileNotFoundError:
            pass
        return dict(states)

    regime_by_config = {}
    for config_name, config_results in all_results.items():
        totals = {"accel_only": 0, "pan_only": 0, "both": 0, "neither": 0, "total": 0}
        for tag, data in config_results.items():
            output_file = os.path.join(output_dir, "ablation", config_name, f"{tag}.txt")
            split, seq = tag.split("_", 1)
            gt_file = os.path.join(data_root, split, seq, "gt", "gt.txt")

            gt_states = _load_track_states(gt_file)
            accel_frames, _ = detect_acceleration_events(gt_states, threshold=accel_thresh)
            warp_mags = data.get("warp_magnitudes", {})
            pan_frames, _ = detect_camera_pan_events(warp_mags, threshold=pan_thresh)

            try:
                from scripts.run_ablation import _get_id_switch_frames
                id_switches = _get_id_switch_frames(gt_file, output_file)
            except Exception:
                id_switches = []

            regime = count_regime_switches(id_switches, accel_frames, pan_frames)
            for k in totals:
                totals[k] += regime[k]

        regime_by_config[config_name] = totals

    print(generate_regime_report(regime_by_config))

    # ── Summary table ──
    print(f"\n{'='*60}")
    print("  ABLATION SUMMARY")
    print(f"{'='*60}")

    header = f"{'Config':<16}{'MOTA':>10}{'IDF1':>10}{'IDS':>8}{'FP':>8}{'FN':>8}{'MT':>6}{'ML':>6}"
    print(header)
    print("-" * len(header))

    for config_name in CONFIGS:
        results = all_results.get(config_name, {})
        evaluated = [r for r in results.values() if "metrics" in r]
        if not evaluated:
            continue
        m = [r["metrics"] for r in evaluated]
        print(f"{config_name:<16}"
              f"{np.mean([x['MOTA'] for x in m]):>10.4f}"
              f"{np.mean([x['IDF1'] for x in m]):>10.4f}"
              f"{sum(x['IDS'] for x in m):>8}"
              f"{sum(x['FP'] for x in m):>8}"
              f"{sum(x['FN'] for x in m):>8}"
              f"{sum(x['MT'] for x in m):>6}"
              f"{sum(x['ML'] for x in m):>6}")

    # Save JSON
    results_file = os.path.join(output_dir, "ablation_results.json")
    serializable = {}
    for cn, seqs in all_results.items():
        serializable[cn] = {}
        for tag, data in seqs.items():
            serializable[cn][tag] = {
                "metrics": data.get("metrics", {}),
                "regime": regime_by_config.get(cn, {}),
            }
    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Saved to {results_file}")

    return all_results


# ── Step 3: Prediction evaluation ────────────────────────────────────────────

def run_prediction_eval(data_root, splits, output_dir):
    """Evaluate KF prediction accuracy (CV vs CA) on all GT sequences."""
    from scripts.evaluate_prediction import evaluate_prediction

    print(f"\n{'='*60}")
    print("[Step 3] PREDICTION ACCURACY (CV 8D vs CA 12D)")
    print(f"{'='*60}")

    horizons = (1, 5, 10, 15, 20)
    all_cv = {h: [] for h in horizons}
    all_ca = {h: [] for h in horizons}
    total_evals = 0
    total_tracks = 0

    for split in splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        for seq in sorted(os.listdir(split_dir)):
            gt_file = os.path.join(split_dir, seq, "gt", "gt.txt")
            if not os.path.exists(gt_file):
                continue

            results, n_eval, n_tracks = evaluate_prediction(
                gt_file, prediction_horizons=horizons,
                min_history=10, eval_interval=5,
            )
            total_evals += n_eval
            total_tracks += n_tracks
            for h in horizons:
                all_cv[h].extend(results["constant_velocity"].get(h, []))
                all_ca[h].extend(results["constant_acceleration"].get(h, []))

    print(f"\n  Aggregated: {total_tracks} tracks, {total_evals} prediction trials\n")

    print(f"{'Horizon':>10} | {'CV Mean (px)':>14} {'CV Med':>10} | "
          f"{'CA Mean (px)':>14} {'CA Med':>10} | {'Improv.':>10}")
    print("-" * 82)

    prediction_summary = {}
    for h in sorted(horizons):
        cv = np.array(all_cv[h])
        ca = np.array(all_ca[h])
        if len(cv) == 0:
            continue
        cv_mean, cv_med = cv.mean(), np.median(cv)
        ca_mean, ca_med = ca.mean(), np.median(ca)
        improv = (cv_mean - ca_mean) / cv_mean * 100

        print(f"{h:>7} fr | {cv_mean:>10.2f} px {cv_med:>10.2f} | "
              f"{ca_mean:>10.2f} px {ca_med:>10.2f} | {improv:>+8.1f}%")

        prediction_summary[h] = {
            "cv_mean": float(cv_mean), "cv_median": float(cv_med),
            "ca_mean": float(ca_mean), "ca_median": float(ca_med),
            "improvement_pct": float(improv),
        }

    pred_file = os.path.join(output_dir, "prediction_results.json")
    with open(pred_file, "w") as f:
        json.dump(prediction_summary, f, indent=2)
    print(f"\n  Saved to {pred_file}")

    return prediction_summary


# ── Step 4: Final comparison ─────────────────────────────────────────────────

def print_final_comparison(ablation_results, _output_dir):
    """Print a unified paper-ready comparison table."""
    print(f"\n{'='*60}")
    print("[Step 4] FINAL COMPARISON TABLE")
    print(f"{'='*60}")

    LABELS = {
        "baseline":   "DeepSORT (baseline)",
        "accel_only": "DeepSORT + CA-KF",
        "cmc_only":   "DeepSORT + ECC-CMC",
        "full":       "DeepSORT + CA-KF + CMC (ours)",
    }

    header = f"{'Method':<32}{'MOTA':>10}{'IDF1':>10}{'IDS':>8}{'FP':>8}{'FN':>8}"
    print(header)
    print("-" * len(header))

    for config_name in ["baseline", "accel_only", "cmc_only", "full"]:
        results = ablation_results.get(config_name, {})
        evaluated = [r["metrics"] for r in results.values() if "metrics" in r]
        if not evaluated:
            continue
        print(f"{LABELS[config_name]:<32}"
              f"{np.mean([x['MOTA'] for x in evaluated]):>10.4f}"
              f"{np.mean([x['IDF1'] for x in evaluated]):>10.4f}"
              f"{sum(x['IDS'] for x in evaluated):>8}"
              f"{sum(x['FP'] for x in evaluated):>8}"
              f"{sum(x['FN'] for x in evaluated):>8}")

    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run complete evaluation pipeline on SportsMOT soccer data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Path to config YAML")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT,
                        help="Dataset root with train/val/test splits (default: soccer/)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Where to write all results")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="Which splits to evaluate (default: train val)")
    parser.add_argument("--yolo_model", default="yolov8x.pt",
                        help="YOLOv8 model for detection generation")
    parser.add_argument("--reid_model", default=None,
                        help="Path to Re-ID model weights (optional)")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="YOLOv8 confidence threshold")

    parser.add_argument("--skip_detections", action="store_true",
                        help="Skip YOLOv8 detection generation")
    parser.add_argument("--skip_ablation", action="store_true",
                        help="Skip four-way ablation")
    parser.add_argument("--skip_prediction", action="store_true",
                        help="Skip KF prediction evaluation")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"  ACCEL-DEEPSORT  —  FULL EVALUATION PIPELINE")
    print(f"{'='*60}")
    print(f"  Config:     {args.config}")
    print(f"  Data root:  {args.data_root}")
    print(f"  Splits:     {args.splits}")
    print(f"  Output:     {args.output_dir}")
    print(f"  YOLO model: {args.yolo_model}")
    print(f"{'='*60}\n")

    t_start = time.time()

    # ── Step 1: Generate detections ──
    if not args.skip_detections:
        generate_missing_detections(
            args.data_root, args.splits,
            yolo_model=args.yolo_model, conf=args.conf,
        )

    # ── Step 2: Ablation ──
    ablation_results = {}
    if not args.skip_ablation:
        ablation_results = run_ablation_all_splits(
            config, args.data_root, args.splits,
            args.output_dir, args.reid_model,
        )

    # ── Step 3: Prediction evaluation ──
    prediction_summary = {}
    if not args.skip_prediction:
        prediction_summary = run_prediction_eval(
            args.data_root, args.splits, args.output_dir,
        )

    # ── Step 4: Final comparison ──
    if ablation_results:
        print_final_comparison(ablation_results, args.output_dir)

    # ── Save combined results ──
    combined = {
        "prediction": prediction_summary,
    }
    combined_file = os.path.join(args.output_dir, "all_results.json")
    with open(combined_file, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    elapsed = time.time() - t_start
    print(f"{'='*60}")
    print(f"  DONE — total time: {elapsed/60:.1f} minutes")
    print(f"  All results saved to: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
