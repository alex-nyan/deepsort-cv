# Evaluation report — accel-deepsort runs

This report summarizes the completed **`scripts/run_all.py`** pipelines you ran locally. Exact commands:

- **SportsMOT (`soccer/`)**: `train` + `val` (30 sequences), detectors generated where missing, **BoT-SORT skipped**.
- **SoccerNet**: `soccernet_data/tracking` **`train`** only (57 sequences), existing **`det/`** used, **BoT-SORT skipped**.

Raw JSON outputs live under `results/soccer_eval/` and `results/soccernet_eval/`. Figures generated from the same files are in `results/analysis/figures/` (run `python scripts/generate_results_figures.py` after edits).

---

## 1. Tracking ablation (DeepSORT variants)

Metrics are **means over sequences** (each sequence contributes one MOTA / IDF1). **IDS** is the **sum** over sequences (total switches).

### SportsMOT soccer — train + val (n = 30)

| Configuration | Mean MOTA | Mean IDF1 | Sum IDS |
|----------------|-----------|-----------|---------|
| Baseline (CV, no CMC) | 0.729 | 0.412 | 8228 |
| Accel only (CA, no CMC) | 0.749 | 0.480 | 4167 |
| CMC only (CV + ECC) | **0.781** | **0.497** | 3808 |
| Full (CA + ECC) | 0.780 | 0.491 | **3639** |

**Takeaways**

- **Camera compensation (ECC)** delivers the largest single jump in MOTA/IDF1 versus baseline.
- **Constant-acceleration Kalman** without CMC improves IDF1 and cuts IDS sharply vs baseline.
- **Full** matches **CMC-only** on MOTA (~0.78) while achieving the **lowest total IDS** — a worthwhile trade-off if identity stability matters more than a tiny MOTA gap.

### SoccerNet tracking — train (n = 57)

| Configuration | Mean MOTA | Mean IDF1 | Sum IDS |
|----------------|-----------|-----------|---------|
| Baseline | 0.893 | 0.556 | 16469 |
| Accel only | 0.917 | 0.629 | 8814 |
| CMC only | **0.947** | **0.741** | **4602** |
| Full | 0.930 | 0.626 | 8319 |

**Takeaways**

- Overall difficulty appears lower than SportsMOT (higher absolute MOTA/IDF1).
- **CMC-only is clearly strongest** on this split for both ranking metrics and IDS.
- **Full underperforms CMC-only** here on IDF1 and IDS — combining CA-KF with ECC **does not always help**; tuning (`configs/defaults.yaml`) or sequence-dependent effects may explain the gap.

See **`fig1_ablation_mota_idf1.png`** and **`fig2_ablation_ids_totals.png`** for bar-chart versions.

---

## 2. Offline Kalman prediction (`prediction_results.json`)

This stage fits **CV vs CA** Kalman filters on **ground-truth tracks** and measures **open-loop** position error at horizons {1, 5, 10, 15, 20} frames (`scripts/evaluate_prediction.py`).

**Observed pattern (both datasets)**

- At **1 frame**, **CA mean error is lower than CV** (~53–55% relative improvement) — short-horizon dynamics benefit from acceleration state.
- From **5 frames onward**, **CA error grows faster than CV** and becomes much worse by 15–20 frames.

That is **expected behavior** for unregularized constant-acceleration models **without continual measurements**: integrating acceleration noise explodes over long horizons, while CV stays bounded in growth rate.

Use **`fig3_prediction_horizon.png`** when explaining that **prediction benchmarks and tracker association behavior are different questions** — tracker CA helps association via filtering under frequent updates; open-loop forecasting is harsh.

---

## 3. What was not run

- **BoT-SORT baseline** (`botsort_results.json` empty): re-run `run_all.py` **without** `--skip_botsort` if you want an external comparison row.
- **SoccerNet test / challenge**: not downloaded/evaluated here.
- **SportsMOT test**: not included in these runs.

---

## 4. Suggested next steps

1. **Tune** on SoccerNet train if “full” should beat “cmc-only” (association thresholds, KF noise, ECC params in `configs/defaults.yaml`).
2. **Run BoT-SORT** once for a complete comparison table in the README sense.
3. **Regenerate figures** after any JSON refresh:

   ```bash
   python scripts/generate_results_figures.py
   ```

   The script sets `MPLCONFIGDIR` under `results/analysis/mpl-cache/` if unset so Matplotlib can write its cache.

**Figure files**

- `results/analysis/figures/fig1_ablation_mota_idf1.png` — mean MOTA & IDF1 by configuration  
- `results/analysis/figures/fig2_ablation_ids_totals.png` — summed IDS  
- `results/analysis/figures/fig3_prediction_horizon.png` — open-loop KF prediction error vs horizon  
