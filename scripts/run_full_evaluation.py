"""
Full evaluation pipeline: runs all experiments described in the paper.

This is a single entry point that:
    1. Runs the four-way DeepSORT ablation
    2. Runs prediction accuracy evaluation
    3. Generates a unified comparison table

Usage:
    python scripts/run_full_evaluation.py \
        --config configs/defaults.yaml \
        --data_root /path/to/dataset \
        --output_dir results/full_eval
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_ablation import run_ablation
from scripts.evaluate_prediction import evaluate_prediction


def run_prediction_evaluation(data_root, output_dir):
    """Run prediction evaluation on all sequences with ground truth."""
    print("\n" + "=" * 60)
    print("PREDICTION ACCURACY (Kalman Filter Motion Models)")
    print("=" * 60)

    sequences = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )

    all_cv_errors = {h: [] for h in [1, 5, 10, 15, 20]}
    all_ca_errors = {h: [] for h in [1, 5, 10, 15, 20]}
    total_evals = 0

    for seq_name in sequences:
        gt_file = os.path.join(data_root, seq_name, "gt", "gt.txt")
        if not os.path.exists(gt_file):
            continue

        results, num_eval, _ = evaluate_prediction(
            gt_file, prediction_horizons=(1, 5, 10, 15, 20),
            min_history=10, eval_interval=5
        )
        total_evals += num_eval

        for h in all_cv_errors:
            all_cv_errors[h].extend(results["constant_velocity"].get(h, []))
            all_ca_errors[h].extend(results["constant_acceleration"].get(h, []))

    # Print aggregated results
    print(f"\nAggregated across {len(sequences)} sequences, {total_evals} trials:")
    print(f"\n{'Horizon':>10} | {'CV (8D) Mean':>14} {'CV Median':>12} | "
          f"{'CA (12D) Mean':>14} {'CA Median':>12} | {'Improvement':>12}")
    print("-" * 85)

    prediction_summary = {}
    for h in sorted(all_cv_errors.keys()):
        cv = np.array(all_cv_errors[h])
        ca = np.array(all_ca_errors[h])
        if len(cv) == 0 or len(ca) == 0:
            continue

        cv_mean, cv_med = cv.mean(), np.median(cv)
        ca_mean, ca_med = ca.mean(), np.median(ca)
        improvement = (cv_mean - ca_mean) / cv_mean * 100

        print(f"{h:>7} fr | {cv_mean:>10.2f} px {cv_med:>10.2f} px | "
              f"{ca_mean:>10.2f} px {ca_med:>10.2f} px | "
              f"{improvement:>+9.1f}%")

        prediction_summary[h] = {
            "cv_mean": float(cv_mean), "cv_median": float(cv_med),
            "ca_mean": float(ca_mean), "ca_median": float(ca_med),
            "improvement_pct": float(improvement),
        }

    return prediction_summary


def generate_final_comparison(ablation_results_file, _output_dir):
    """Generate the final paper-ready comparison table."""
    print("\n" + "=" * 60)
    print("FINAL COMPARISON TABLE (For Paper)")
    print("=" * 60)

    # Load ablation results
    ablation_data = {}
    if os.path.exists(ablation_results_file):
        with open(ablation_results_file) as f:
            ablation_data = json.load(f)

    header = f"{'Method':<30}{'MOTA':>10}{'IDF1':>10}{'IDS':>8}{'FP':>8}{'FN':>8}"
    print(header)
    print("-" * len(header))

    # Print ablation results
    for config_name in ["baseline", "accel_only", "cmc_only", "full"]:
        if config_name not in ablation_data:
            continue
        seqs = ablation_data[config_name]
        metrics_list = [s["metrics"] for s in seqs.values() if "metrics" in s and s["metrics"]]
        if not metrics_list:
            continue

        avg_mota = np.mean([m["MOTA"] for m in metrics_list])
        avg_idf1 = np.mean([m["IDF1"] for m in metrics_list])
        total_ids = sum(m["IDS"] for m in metrics_list)
        total_fp = sum(m["FP"] for m in metrics_list)
        total_fn = sum(m["FN"] for m in metrics_list)

        label = {
            "baseline": "DeepSORT (baseline)",
            "accel_only": "DeepSORT + CA-KF",
            "cmc_only": "DeepSORT + ECC-CMC",
            "full": "DeepSORT + CA-KF + CMC (ours)",
        }[config_name]

        print(f"{label:<30}{avg_mota:>10.4f}{avg_idf1:>10.4f}"
              f"{total_ids:>8}{total_fp:>8}{total_fn:>8}")


def main():
    parser = argparse.ArgumentParser(description="Full evaluation pipeline")
    parser.add_argument("--config", default="configs/defaults.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default="results/full_eval")
    parser.add_argument("--reid_model", default=None)
    parser.add_argument("--skip_ablation", action="store_true")
    parser.add_argument("--skip_prediction", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Four-way ablation ---
    if not args.skip_ablation:
        run_ablation(config, args.data_root, args.output_dir, args.reid_model)

    # --- 2. Prediction evaluation ---
    prediction_summary = {}
    if not args.skip_prediction:
        prediction_summary = run_prediction_evaluation(args.data_root, args.output_dir)

    # --- 3. Final comparison ---
    ablation_results_file = os.path.join(args.output_dir, "ablation_results.json")
    generate_final_comparison(ablation_results_file, args.output_dir)

    # Save all results
    all_results = {
        "prediction": prediction_summary,
    }
    with open(os.path.join(args.output_dir, "full_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
