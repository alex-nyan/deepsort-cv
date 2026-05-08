"""
Sweep CA process-noise (std_weight_acceleration) with ECC enabled on a dataset.

Produces a JSON mapping each noise value to per-sequence MOT metrics so we can
show that the Full-config regression on SoccerNet is tuning-dependent, not a
fundamental flaw of the CA model.

The CMC-only (CV+ECC) reference is computed once as a baseline for the plot.

Usage:
    # Default: sweep on SoccerNet train
    python scripts/run_ca_sensitivity.py

    # Custom sweep values / data
    python scripts/run_ca_sensitivity.py \
        --data_root soccernet_data/tracking \
        --splits train \
        --sweep 0.005 0.01 0.025 0.05 0.1 \
        --output_dir results/ca_sensitivity
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "defaults.yaml")
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "soccernet_data", "tracking")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "ca_sensitivity")

DEFAULT_SWEEP = [5e-3, 1e-2, 2.5e-2, 5e-2, 1e-1]


def discover_sequences(data_root, splits):
    """Return list of (split, seq_name, seq_dir) with both det.txt and gt.txt."""
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
    return sequences


def run_config_on_sequences(config, sequences, output_dir, tag,
                            kalman_model, cmc, kf_override=None):
    """
    Run one tracker configuration over all sequences and return per-sequence metrics.
    """
    from scripts.run_tracker import run_tracker
    from evaluation.mot_metrics import evaluate_sequence

    cfg = copy.deepcopy(config)
    if kf_override:
        for k, v in kf_override.items():
            cfg["kalman"]["ca"][k] = v

    run_dir = os.path.join(output_dir, tag)
    os.makedirs(run_dir, exist_ok=True)

    per_seq = {}

    for split, seq, seq_dir in sequences:
        seq_tag = f"{split}_{seq}"
        output_file = os.path.join(run_dir, f"{seq_tag}.txt")

        t0 = time.time()
        run_tracker(
            cfg, seq_dir, output_file,
            kalman_model=kalman_model,
            camera_motion_compensation=cmc,
        )
        gt_file = os.path.join(seq_dir, "gt", "gt.txt")
        metrics = evaluate_sequence(gt_file, output_file)
        elapsed = time.time() - t0

        per_seq[seq_tag] = metrics
        print(f"    [{seq_tag}] MOTA={metrics['MOTA']:.4f}  "
              f"IDF1={metrics['IDF1']:.4f}  IDS={metrics['IDS']}  "
              f"({elapsed:.1f}s)")

    return per_seq


def summarise(per_seq):
    """Aggregate per-sequence metrics into mean MOTA, mean IDF1, sum IDS."""
    m = list(per_seq.values())
    return {
        "MOTA_mean": float(np.mean([x["MOTA"] for x in m])),
        "IDF1_mean": float(np.mean([x["IDF1"] for x in m])),
        "IDS_sum": int(sum(x["IDS"] for x in m)),
        "MOTA_std": float(np.std([x["MOTA"] for x in m])),
        "IDF1_std": float(np.std([x["IDF1"] for x in m])),
        "n": len(m),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sweep CA process-noise with ECC on a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sweep", nargs="+", type=float, default=DEFAULT_SWEEP,
        help="std_weight_acceleration values to sweep (default: %(default)s)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    sequences = discover_sequences(args.data_root, args.splits)
    if not sequences:
        print("No valid sequences found. Exiting.")
        return

    print(f"{'='*60}")
    print(f"  CA PROCESS-NOISE SENSITIVITY SWEEP")
    print(f"{'='*60}")
    print(f"  Data root:  {args.data_root}")
    print(f"  Splits:     {args.splits}")
    print(f"  Sequences:  {len(sequences)}")
    print(f"  Sweep:      {args.sweep}")
    print(f"  Output:     {args.output_dir}")
    print(f"{'='*60}\n")

    results = {}

    # ── Reference: CMC-only (CV + ECC) ────────────────────────────────
    print("  [Reference] CMC-only (CV + ECC)")
    cmc_per_seq = run_config_on_sequences(
        config, sequences, args.output_dir,
        tag="cmc_only",
        kalman_model="constant_velocity",
        cmc=True,
    )
    cmc_summary = summarise(cmc_per_seq)
    results["cmc_only"] = {
        "summary": cmc_summary,
        "per_sequence": cmc_per_seq,
    }
    print(f"\n  CMC-only  =>  MOTA={cmc_summary['MOTA_mean']:.4f}  "
          f"IDF1={cmc_summary['IDF1_mean']:.4f}  "
          f"IDS={cmc_summary['IDS_sum']}\n")

    # ── Sweep: Full (CA + ECC) at each noise value ────────────────────
    results["sweep"] = {}

    for accel_noise in sorted(args.sweep):
        label = f"ca_accel_{accel_noise:.0e}"
        print(f"  [Sweep] std_weight_acceleration = {accel_noise}")
        sweep_per_seq = run_config_on_sequences(
            config, sequences, args.output_dir,
            tag=label,
            kalman_model="constant_acceleration",
            cmc=True,
            kf_override={"std_weight_acceleration": accel_noise},
        )
        s = summarise(sweep_per_seq)
        results["sweep"][str(accel_noise)] = {
            "summary": s,
            "per_sequence": sweep_per_seq,
        }
        print(f"\n  accel={accel_noise:.0e}  =>  MOTA={s['MOTA_mean']:.4f}  "
              f"IDF1={s['IDF1_mean']:.4f}  IDS={s['IDS_sum']}\n")

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SENSITIVITY SWEEP SUMMARY")
    print(f"{'='*60}")
    header = f"{'accel_noise':>14}  {'MOTA':>8}  {'IDF1':>8}  {'IDS':>6}  {'note':>10}"
    print(header)
    print("-" * len(header))

    print(f"{'CMC-only':>14}  "
          f"{cmc_summary['MOTA_mean']:>8.4f}  "
          f"{cmc_summary['IDF1_mean']:>8.4f}  "
          f"{cmc_summary['IDS_sum']:>6}  "
          f"{'reference':>10}")

    for accel_noise in sorted(args.sweep):
        s = results["sweep"][str(accel_noise)]["summary"]
        delta_idf1 = s["IDF1_mean"] - cmc_summary["IDF1_mean"]
        marker = "***" if delta_idf1 >= 0 else ""
        print(f"{accel_noise:>14.0e}  "
              f"{s['MOTA_mean']:>8.4f}  "
              f"{s['IDF1_mean']:>8.4f}  "
              f"{s['IDS_sum']:>6}  "
              f"{delta_idf1:>+8.4f} {marker}")

    # ── Save JSON ─────────────────────────────────────────────────────
    out_file = os.path.join(args.output_dir, "sensitivity_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {out_file}")


if __name__ == "__main__":
    main()
