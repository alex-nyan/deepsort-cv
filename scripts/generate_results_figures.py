#!/usr/bin/env python3
"""
Load results/soccer_eval and results/soccernet_eval JSON summaries and write
publication-style figures to results/analysis/figures/.

Requires: matplotlib (pip install matplotlib)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT / "results" / "analysis" / "figures"
# Writable config dir (avoids ~/.matplotlib permission issues in sandboxes / CI)
_MPL = PROJECT / "results" / "analysis" / "mpl-cache"
_MPL.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL))
os.environ.setdefault("MPLBACKEND", "Agg")


def load_ablation(path: Path):
    with open(path) as f:
        return json.load(f)


def aggregate_ablation(data: dict):
    ORDER = ["baseline", "accel_only", "cmc_only", "full"]
    out = {}
    for cfg in ORDER:
        seqs = data.get(cfg, {})
        mota, idf1, ids = [], [], []
        for _, payload in seqs.items():
            m = payload.get("metrics") or {}
            if not m:
                continue
            mota.append(m.get("MOTA", 0))
            idf1.append(m.get("IDF1", 0))
            ids.append(m.get("IDS", 0))
        if mota:
            out[cfg] = {
                "MOTA_mean": np.mean(mota),
                "IDF1_mean": np.mean(idf1),
                "IDS_sum": int(np.sum(ids)),
                "n": len(mota),
            }
    return out


def main():
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except ImportError:
        print("Install matplotlib: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )

    soccer_path = PROJECT / "results" / "soccer_eval" / "ablation_results.json"
    sn_path = PROJECT / "results" / "soccernet_eval" / "ablation_results.json"
    pred_soccer = PROJECT / "results" / "soccer_eval" / "prediction_results.json"
    pred_sn = PROJECT / "results" / "soccernet_eval" / "prediction_results.json"

    soccer = aggregate_ablation(load_ablation(soccer_path))
    soccernet = aggregate_ablation(load_ablation(sn_path))

    ORDER = ["baseline", "accel_only", "cmc_only", "full"]
    LABELS = ["Baseline\n(CV)", "Accel only\n(CA)", "CMC only\n(CV+ECC)", "Full\n(CA+ECC)"]
    x = np.arange(len(ORDER))
    width = 0.35

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: MOTA & IDF1 ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    def plot_metrics(ax, agg, title):
        m_mota = [agg[c]["MOTA_mean"] for c in ORDER]
        m_idf = [agg[c]["IDF1_mean"] for c in ORDER]
        ax.bar(x - width / 2, m_mota, width, label="MOTA (mean)", color="#4C72B0")
        ax.bar(x + width / 2, m_idf, width, label="IDF1 (mean)", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=9)

    plot_metrics(axes[0], soccer, "SportsMOT soccer (train + val, n=30 seq)")
    plot_metrics(axes[1], soccernet, "SoccerNet tracking train (n=57 seq)")
    fig.suptitle("Four-way ablation: mean MOTA & IDF1 per sequence", y=1.02)
    fig.savefig(OUT_DIR / "fig1_ablation_mota_idf1.png")

    # ── Figure 2: Identity switches (totals) ──────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ids_soccer = [soccer[c]["IDS_sum"] for c in ORDER]
    ids_sn = [soccernet[c]["IDS_sum"] for c in ORDER]
    ax2.barh(x - width / 2, ids_soccer, width, label="SportsMOT soccer", color="#55A868")
    ax2.barh(x + width / 2, ids_sn, width, label="SoccerNet train", color="#C44E52")
    ax2.set_yticks(x)
    ax2.set_yticklabels(["Baseline", "Accel only", "CMC only", "Full"])
    ax2.set_xlabel("Total IDS (summed over all sequences)")
    ax2.set_title("Identity switches by configuration")
    ax2.legend()
    fig2.savefig(OUT_DIR / "fig2_ablation_ids_totals.png")
    plt.close(fig2)

    # ── Figure 3: Offline KF prediction error ────────────────────────
    with open(pred_soccer) as f:
        ps = json.load(f)
    with open(pred_sn) as f:
        pn = json.load(f)

    horizons = sorted(int(k) for k in ps.keys())
    cv_s = [ps[str(h)]["cv_mean"] for h in horizons]
    ca_s = [ps[str(h)]["ca_mean"] for h in horizons]
    cv_n = [pn[str(h)]["cv_mean"] for h in horizons]
    ca_n = [pn[str(h)]["ca_mean"] for h in horizons]

    fig3, (ax31, ax32) = plt.subplots(1, 2, figsize=(10, 4))
    ax31.plot(horizons, cv_s, "o-", label="CV (8D)", color="#4C72B0")
    ax31.plot(horizons, ca_s, "s-", label="CA (12D)", color="#C44E52")
    ax31.set_xlabel("Prediction horizon (frames)")
    ax31.set_ylabel("Mean pixel error")
    ax31.set_title("SportsMOT soccer")
    ax31.legend()
    ax31.grid(alpha=0.3)

    ax32.plot(horizons, cv_n, "o-", label="CV (8D)", color="#4C72B0")
    ax32.plot(horizons, ca_n, "s-", label="CA (12D)", color="#C44E52")
    ax32.set_xlabel("Prediction horizon (frames)")
    ax32.set_ylabel("Mean pixel error")
    ax32.set_title("SoccerNet train")
    ax32.legend()
    ax32.grid(alpha=0.3)

    fig3.suptitle(
        "Open-loop Kalman prediction from GT tracks (evaluate_prediction.py)",
        y=1.02,
    )
    fig3.savefig(OUT_DIR / "fig3_prediction_horizon.png")
    plt.close(fig3)
    plt.close(fig)

    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
