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
from scipy.stats import wilcoxon

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
    """Return per-config aggregates *and* per-sequence arrays (keyed by seq name)."""
    ORDER = ["baseline", "accel_only", "cmc_only", "full"]
    out = {}
    for cfg in ORDER:
        seqs = data.get(cfg, {})
        mota, idf1, ids, seq_names = [], [], [], []
        for seq_name, payload in sorted(seqs.items()):
            m = payload.get("metrics") or {}
            if not m:
                continue
            mota.append(m.get("MOTA", 0))
            idf1.append(m.get("IDF1", 0))
            ids.append(m.get("IDS", 0))
            seq_names.append(seq_name)
        if mota:
            mota_arr = np.array(mota)
            idf1_arr = np.array(idf1)
            n = len(mota_arr)
            out[cfg] = {
                "MOTA_mean": float(np.mean(mota_arr)),
                "IDF1_mean": float(np.mean(idf1_arr)),
                "MOTA_std": float(np.std(mota_arr, ddof=1)),
                "IDF1_std": float(np.std(idf1_arr, ddof=1)),
                "MOTA_ci95": float(1.96 * np.std(mota_arr, ddof=1) / np.sqrt(n)),
                "IDF1_ci95": float(1.96 * np.std(idf1_arr, ddof=1) / np.sqrt(n)),
                "IDS_sum": int(np.sum(ids)),
                "n": n,
                "MOTA_per_seq": mota_arr,
                "IDF1_per_seq": idf1_arr,
                "IDS_per_seq": np.array(ids),
                "seq_names": seq_names,
            }
    return out


def pairwise_significance(agg: dict, metric: str = "MOTA"):
    """Paired Wilcoxon signed-rank tests between all config pairs.

    Returns list of (cfg_a, cfg_b, stat, p_value) tuples.
    Only meaningful when sequences are aligned (same set, same order).
    """
    ORDER = ["baseline", "accel_only", "cmc_only", "full"]
    key = f"{metric}_per_seq"
    results = []
    for i, a in enumerate(ORDER):
        for b in ORDER[i + 1:]:
            if a not in agg or b not in agg:
                continue
            va, vb = agg[a][key], agg[b][key]
            if len(va) != len(vb) or len(va) < 6:
                continue
            if np.array_equal(va, vb):
                results.append((a, b, 0.0, 1.0))
                continue
            stat, p = wilcoxon(va, vb, alternative="two-sided")
            results.append((a, b, float(stat), float(p)))
    return results


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

    # ── Figure 1: MOTA & IDF1 (with 95% CI error bars) ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    def plot_metrics(ax, agg, title):
        m_mota = [agg[c]["MOTA_mean"] for c in ORDER]
        m_idf = [agg[c]["IDF1_mean"] for c in ORDER]
        ci_mota = [agg[c]["MOTA_ci95"] for c in ORDER]
        ci_idf = [agg[c]["IDF1_ci95"] for c in ORDER]
        ax.bar(x - width / 2, m_mota, width, yerr=ci_mota,
               capsize=3, label="MOTA (mean)", color="#4C72B0", ecolor="#2B4570")
        ax.bar(x + width / 2, m_idf, width, yerr=ci_idf,
               capsize=3, label="IDF1 (mean)", color="#DD8452", ecolor="#A85A2A")
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=9)

    plot_metrics(axes[0], soccer, "SportsMOT soccer (n=30 seq)")
    plot_metrics(axes[1], soccernet, "SoccerNet tracking train (n=57 seq)")
    fig.suptitle("Four-way ablation: mean MOTA & IDF1 per sequence (±95% CI)", y=1.02)
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

    # ── Figure 4: CA process-noise sensitivity ────────────────────────
    sens_path = PROJECT / "results" / "ca_sensitivity" / "sensitivity_results.json"
    if sens_path.exists():
        with open(sens_path) as f:
            sens = json.load(f)

        cmc_ref = sens["cmc_only"]["summary"]
        sweep = sens["sweep"]
        noise_vals = sorted(float(k) for k in sweep.keys())
        mota_vals = [sweep[str(n)]["summary"]["MOTA_mean"] for n in noise_vals]
        idf1_vals = [sweep[str(n)]["summary"]["IDF1_mean"] for n in noise_vals]
        ids_vals = [sweep[str(n)]["summary"]["IDS_sum"] for n in noise_vals]

        fig4, (ax_idf1, ax_ids) = plt.subplots(1, 2, figsize=(11, 4.2))

        ax_idf1.plot(range(len(noise_vals)), mota_vals, "o-",
                     label="MOTA (Full)", color="#4C72B0")
        ax_idf1.plot(range(len(noise_vals)), idf1_vals, "s-",
                     label="IDF1 (Full)", color="#DD8452")
        ax_idf1.axhline(cmc_ref["MOTA_mean"], ls="--", color="#4C72B0",
                        alpha=0.6, label="MOTA (CMC-only ref)")
        ax_idf1.axhline(cmc_ref["IDF1_mean"], ls="--", color="#DD8452",
                        alpha=0.6, label="IDF1 (CMC-only ref)")
        ax_idf1.set_xticks(range(len(noise_vals)))
        ax_idf1.set_xticklabels([f"{v:.0e}" for v in noise_vals], fontsize=8)
        ax_idf1.set_xlabel("std_weight_acceleration")
        ax_idf1.set_ylabel("Score")
        ax_idf1.set_title("MOTA / IDF1 vs CA process noise")
        ax_idf1.legend(fontsize=8, loc="lower left")
        ax_idf1.grid(alpha=0.3)

        ax_ids.plot(range(len(noise_vals)), ids_vals, "D-",
                    label="IDS (Full)", color="#C44E52")
        ax_ids.axhline(cmc_ref.get("IDS_sum", 0), ls="--", color="#C44E52",
                       alpha=0.6, label="IDS (CMC-only ref)")
        ax_ids.set_xticks(range(len(noise_vals)))
        ax_ids.set_xticklabels([f"{v:.0e}" for v in noise_vals], fontsize=8)
        ax_ids.set_xlabel("std_weight_acceleration")
        ax_ids.set_ylabel("Total identity switches")
        ax_ids.set_title("IDS vs CA process noise")
        ax_ids.legend(fontsize=8)
        ax_ids.grid(alpha=0.3)

        n_seq = cmc_ref.get("n", "?")
        fig4.suptitle(
            f"CA process-noise sensitivity (SoccerNet train, n={n_seq} seq)",
            y=1.02,
        )
        fig4.savefig(OUT_DIR / "fig4_ca_sensitivity.png")
        plt.close(fig4)
        print(f"  fig4_ca_sensitivity.png written")
    else:
        print(f"  [skip] {sens_path} not found — run scripts/run_ca_sensitivity.py first")

    # ── Figure 5: Per-sequence box plots ─────────────────────────────
    fig5, axes5 = plt.subplots(2, 2, figsize=(12, 8))

    def draw_boxplot(ax, agg, metric, title):
        key = f"{metric}_per_seq"
        bp_data = [agg[c][key] for c in ORDER if c in agg]
        bp_labels = [LABELS[i] for i, c in enumerate(ORDER) if c in agg]
        bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=1.5))
        colors = ["#8DA0CB", "#A6D854", "#FFD92F", "#FC8D62"]
        for patch, color in zip(bp["boxes"], colors[:len(bp_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    draw_boxplot(axes5[0, 0], soccer, "MOTA", "SportsMOT — MOTA per sequence")
    draw_boxplot(axes5[0, 1], soccer, "IDF1", "SportsMOT — IDF1 per sequence")
    draw_boxplot(axes5[1, 0], soccernet, "MOTA", "SoccerNet — MOTA per sequence")
    draw_boxplot(axes5[1, 1], soccernet, "IDF1", "SoccerNet — IDF1 per sequence")

    fig5.suptitle("Per-sequence metric distributions", y=1.01)
    fig5.tight_layout()
    fig5.savefig(OUT_DIR / "fig5_per_sequence_boxplots.png")
    plt.close(fig5)
    print("  fig5_per_sequence_boxplots.png written")

    # ── Statistical significance (paired Wilcoxon) ────────────────────
    print(f"\n{'='*70}")
    print("  PAIRED WILCOXON SIGNED-RANK TESTS")
    print(f"{'='*70}")

    for dataset_name, agg in [("SportsMOT", soccer), ("SoccerNet", soccernet)]:
        print(f"\n  {dataset_name}:")
        header = f"    {'Pair':<30}  {'MOTA p':>10}  {'IDF1 p':>10}  {'sig?':>6}"
        print(header)
        print("    " + "-" * (len(header) - 4))

        mota_tests = pairwise_significance(agg, "MOTA")
        idf1_tests = pairwise_significance(agg, "IDF1")
        idf1_map = {(a, b): p for a, b, _, p in idf1_tests}

        for a, b, _, p_mota in mota_tests:
            p_idf1 = idf1_map.get((a, b), float("nan"))
            sig = "*" if (p_mota < 0.05 or p_idf1 < 0.05) else ""
            print(f"    {a + ' vs ' + b:<30}  {p_mota:>10.4f}  {p_idf1:>10.4f}  {sig:>6}")

    # Save significance results alongside figures
    sig_results = {}
    for dataset_name, agg in [("SportsMOT", soccer), ("SoccerNet", soccernet)]:
        sig_results[dataset_name] = {}
        for metric in ("MOTA", "IDF1"):
            tests = pairwise_significance(agg, metric)
            sig_results[dataset_name][metric] = [
                {"pair": f"{a} vs {b}", "statistic": s, "p_value": p}
                for a, b, s, p in tests
            ]
    sig_file = OUT_DIR / "significance_tests.json"
    with open(sig_file, "w") as f:
        json.dump(sig_results, f, indent=2)
    print(f"\n  Saved to {sig_file}")

    print(f"\nWrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
