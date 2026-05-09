# Acceleration-Aware DeepSORT with Camera-Motion Compensation

Multi-object tracking for broadcast soccer, extending DeepSORT with:
1. **Constant-acceleration Kalman filter** (12D state vector) for bursty player dynamics
2. **ECC camera-motion compensation** to correct for broadcast pan/zoom

The repository ships **evaluation summaries, sensitivity outputs, publication figures, and LaTeX** so you can verify or extend the experiments without rerunning trackers. Intermediate per-sequence MOT outputs (`*_eval/ablation/*/...txt`, CA-sensitivity sweep folders under `results/ca_sensitivity/`) are intentionally **not** tracked; running `scripts/run_all.py` or `scripts/run_ca_sensitivity.py` regenerates them under `results/`.

## Project Structure

```
accel-deepsort/
├── deep_sort/
│   ├── sort/
│   │   ├── kalman_filter.py          # Original 8D constant-velocity KF
│   │   ├── kalman_filter_accel.py    # 12D constant-acceleration KF
│   │   ├── track.py                  # Single-target track state machine
│   │   ├── tracker.py                # Multi-target tracker (predict→associate loop)
│   │   ├── nn_matching.py            # Nearest-neighbor matching (cosine/euclidean)
│   │   ├── linear_assignment.py      # Hungarian + cascade matching
│   │   ├── iou_matching.py           # IoU-based fallback matching
│   │   ├── detection.py              # Detection data class
│   │   └── preprocessing.py          # NMS and utilities
│   └── reid/
│       └── extractor.py              # Re-ID feature extraction (disabled in ablation)
├── camera_motion/
│   └── ecc_compensator.py            # ECC affine registration module
├── evaluation/
│   ├── mot_metrics.py                # MOTA, IDF1, ID-switch computation (py-motmetrics)
│   └── regime_analysis.py            # Per-regime IDS breakdown (accel/pan events)
├── configs/
│   └── defaults.yaml                 # All hyperparameters
├── scripts/
│   ├── run_all.py                    # Full pipeline: detections → ablation → prediction
│   ├── run_tracker.py                # Run one tracker config on one sequence
│   ├── run_ablation.py               # Four-way ablation driver (+ regime analysis)
│   ├── run_full_evaluation.py        # Alternative pipeline entry point
│   ├── run_ca_sensitivity.py         # CA process-noise sweep (σ_acc sensitivity)
│   ├── evaluate_prediction.py        # KF open-loop prediction accuracy (CV vs CA)
│   └── generate_results_figures.py   # All paper figures + statistical tests
├── tools/
│   ├── generate_detections.py        # YOLOv8 detection generation → det/det.txt
│   ├── convert_soccernet.py          # SoccerNet JSON → MOTChallenge format
│   └── convert_sportsmot.py          # SportsMOT → MOTChallenge format (soccer filter)
├── results/
│   ├── soccer_eval/                  # Aggregates (tracked): ablation + prediction JSON
│   ├── soccernet_eval/               # Same for SoccerNet train
│   ├── ca_sensitivity/               # sensitivity_results.json (tracked); per-run dirs local only
│   └── analysis/
│       ├── figures/                  # Publication PNGs + significance_tests.json
│       ├── cvpr_results_figures.tex  # LaTeX experiments section
│       └── EVALUATION_REPORT.md      # Written summary of runs and figures
└── requirements.txt
```

## Prerequisites

- **Python 3.10+** (for example 3.11 or 3.12; avoid pre-release interpreters for dependencies)
- **GPU recommended** for YOLOv8 detection generation (CPU works but is slow)

## Setup

```bash
# Clone and enter the project
git clone <repo-url> accel-deepsort
cd accel-deepsort

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24 | Array operations |
| scipy | ≥1.10 | Wilcoxon signed-rank tests |
| opencv-python | ≥4.8 | ECC registration, image I/O |
| torch | ≥2.0 | Backend for YOLOv8 |
| torchvision | ≥0.15 | Backend for YOLOv8 |
| ultralytics | ≥8.0 | YOLOv8 detection generation |
| motmetrics | ≥1.4 | MOTChallenge metrics (MOTA, IDF1, IDS) |
| PyYAML | ≥6.0 | Config loading |
| tqdm | ≥4.65 | Progress bars |
| filterpy | ≥1.4 | Kalman filter utilities |
| matplotlib | ≥3.7 | Figure generation |

## Datasets

Two datasets are used. Both must be in MOTChallenge directory layout:

```
dataset_root/
├── train/
│   ├── SEQUENCE_001/
│   │   ├── img1/          # Frame images (000001.jpg, 000002.jpg, ...)
│   │   ├── gt/gt.txt      # Ground-truth annotations
│   │   └── det/det.txt    # Detections (generated or provided)
│   ├── SEQUENCE_002/
│   │   └── ...
├── val/
│   └── ...
```

### Dataset A: SportsMOT (soccer subset)

1. Download SportsMOT from [https://github.com/MCG-NJU/SportsMOT](https://github.com/MCG-NJU/SportsMOT) and extract to `sportsmot_publish/`.

2. Convert to MOTChallenge layout, filtering for soccer sequences only:

```bash
python tools/convert_sportsmot.py \
    --sportsmot_root sportsmot_publish/dataset \
    --output_dir soccer \
    --sport soccer
```

This creates `soccer/train/` (15 sequences), `soccer/val/` (15 sequences), and `soccer/test/`.

3. Generate person detections with YOLOv8 for sequences that lack `det/det.txt`:

```bash
python tools/generate_detections.py --data_root soccer/train --model yolov8x.pt --conf 0.3
python tools/generate_detections.py --data_root soccer/val   --model yolov8x.pt --conf 0.3
```

YOLOv8x weights are downloaded automatically on first run. Detections are written as `det/det.txt` inside each sequence folder, in MOTChallenge format: `frame,-1,x,y,w,h,conf,-1,-1,-1`.

### Dataset B: SoccerNet-Tracking

1. Download SoccerNet-Tracking from [https://www.soccer-net.org/](https://www.soccer-net.org/) (requires registration).

2. Convert to MOTChallenge layout:

```bash
python tools/convert_soccernet.py \
    --soccernet_root /path/to/soccernet \
    --output_dir soccernet_data
```

This creates `soccernet_data/tracking/train/` (57 sequences). SoccerNet provides its own annotation-derived detections, so detection generation is not required.

## Reproducing All Results

The full replication pipeline consists of 6 steps. Steps 1–3 are the primary experiments; steps 4–6 produce figures, tables, and statistical tests for the paper.

### Step 1: SportsMOT Evaluation (four-way ablation + prediction)

Runs the complete pipeline on SportsMOT soccer (train + val, 30 sequences):

```bash
python scripts/run_all.py \
    --config configs/defaults.yaml \
    --data_root soccer \
    --output_dir results/soccer_eval \
    --splits train val \
    --yolo_model yolov8x.pt
```

This executes 4 sub-steps automatically:
1. **Detection generation** — creates `det/det.txt` for any sequence missing one (YOLOv8x, conf=0.3, 1280px)
2. **Four-way ablation** — runs Baseline, Accel-only, CMC-only, Full on all sequences
3. **Regime analysis** — classifies IDS by cause (acceleration event, camera pan, both, neither)
4. **KF prediction evaluation** — compares CV (8D) vs CA (12D) open-loop prediction at horizons 1/5/10/15/20 frames

**Outputs (aggregates tracked in git; per-sequence MOT `.txt` files live under `results/soccer_eval/ablation/` locally and are gitignored):**
- `results/soccer_eval/ablation_results.json` — per-sequence metrics for all 4 configs
- `results/soccer_eval/prediction_results.json` — CV vs CA prediction errors
- `results/soccer_eval/all_results.json` — combined prediction summary

**Skip flags** (for partial re-runs):
```bash
--skip_detections    # det/det.txt files already exist
--skip_ablation      # skip four-way ablation
--skip_prediction    # skip KF prediction evaluation
```

**Expected runtime:** ~2–6 hours on a modern GPU (mostly detection generation).

### Step 2: SoccerNet Evaluation (four-way ablation + prediction)

Runs the same ablation and prediction evaluation on SoccerNet-Tracking train (57 sequences):

```bash
python scripts/run_all.py \
    --config configs/defaults.yaml \
    --data_root soccernet_data/tracking \
    --output_dir results/soccernet_eval \
    --splits train \
    --skip_detections
```

We skip detection generation (SoccerNet provides detections).

**Outputs:** same layout as SportsMOT (`ablation_results.json`, `prediction_results.json`, `all_results.json`) plus local-only `ablation/<config>/*.txt` trajectories.

**Expected runtime:** ~2–4 hours (tracking only, no detection generation).

### Step 3: CA Process-Noise Sensitivity Sweep

Sweeps the acceleration process-noise parameter σ_acc with ECC enabled on SoccerNet, demonstrating that the Full-config regression is a tuning artifact:

```bash
python scripts/run_ca_sensitivity.py \
    --config configs/defaults.yaml \
    --data_root soccernet_data/tracking \
    --splits train \
    --sweep 0.005 0.025 0.1 \
    --output_dir results/ca_sensitivity
```

This runs 4 full evaluations over 57 sequences (1× CMC-only reference + 3× CA+ECC at different noise values).

**Outputs:**
- `results/ca_sensitivity/sensitivity_results.json` — per-sequence metrics for each noise value (the file in git matches the paper sweep; rerunning creates `cmc_only/` and `ca_accel_*` directories locally—those are gitignored)

**Expected runtime:** ~3–6 hours.

### Step 4: Generate Paper Figures

Once Steps 1–3 are complete, generate all publication figures and statistical tests:

```bash
python scripts/generate_results_figures.py
```

**Important:** This script has **hardcoded input paths** — it reads from `results/soccer_eval/ablation_results.json`, `results/soccernet_eval/ablation_results.json`, and their corresponding `prediction_results.json` files. The `--output_dir` values in Steps 1–2 must match these paths exactly (as shown above). If you used different output directories, either re-run with the correct paths or symlink/copy the JSON files.

**Outputs:**

| File | Description |
|------|-------------|
| `fig1_ablation_mota_idf1.png` | Four-way ablation MOTA & IDF1 bar chart (±95% CI) |
| `fig2_ablation_ids_totals.png` | Identity switches by configuration (horizontal bars) |
| `fig3_prediction_horizon.png` | Open-loop KF prediction error vs horizon (CV vs CA) |
| `fig4_ca_sensitivity.png` | CA process-noise sensitivity (MOTA/IDF1/IDS vs σ_acc) |
| `fig5_per_sequence_boxplots.png` | Per-sequence metric distributions (box plots) |
| `significance_tests.json` | Paired Wilcoxon signed-rank test p-values |

### Step 5: Compile LaTeX Report

The experiments section and all figure/table references are in:

```
results/analysis/cvpr_results_figures.tex
```

To compile, ensure your figures are accessible and run:

```bash
# From the results/analysis directory (or set \graphicspath accordingly)
pdflatex cvpr_results_figures.tex
```

### Step 6: Verify Results

Check that all expected output files exist:

```bash
# Core ablation results (both datasets)
ls results/soccer_eval/ablation_results.json
ls results/soccernet_eval/ablation_results.json

# Prediction evaluation
ls results/soccer_eval/prediction_results.json
ls results/soccernet_eval/prediction_results.json

# Sensitivity sweep
ls results/ca_sensitivity/sensitivity_results.json

# Figures
ls results/analysis/figures/fig{1,2,3,4,5}_*.png
ls results/analysis/figures/significance_tests.json
```

## Running Individual Components

### Single Sequence Tracking

```bash
python scripts/run_tracker.py \
    --config configs/defaults.yaml \
    --sequence soccer/train/v_1yHWGw8DH4A_c029 \
    --output results/track.txt \
    --kalman_model constant_acceleration \
    --camera_motion_compensation
```

Options for `--kalman_model`: `constant_velocity` (8D) or `constant_acceleration` (12D).
Add `--camera_motion_compensation` to enable ECC.

### Four-Way Ablation Only

```bash
python scripts/run_ablation.py \
    --config configs/defaults.yaml \
    --data_root soccer/train \
    --output_dir results/ablation
```

### KF Prediction Evaluation Only

```bash
python scripts/evaluate_prediction.py \
    --gt_file soccer/train/v_1yHWGw8DH4A_c029/gt/gt.txt
```

### Detection Generation Only

```bash
python tools/generate_detections.py \
    --data_root soccer/train \
    --model yolov8x.pt \
    --conf 0.3
```

## Ablation Configurations

The four configurations are a 2×2 factorial crossing two binary factors:

| Config | Kalman Filter | Camera Compensation | Description |
|--------|--------------|---------------------|-------------|
| `baseline` | CV (8D) | None | Original DeepSORT |
| `accel_only` | CA (12D) | None | Acceleration-aware KF only |
| `cmc_only` | CV (8D) | ECC affine | Camera compensation only |
| `full` | CA (12D) | ECC affine | Both modifications |

All four share the same detections, the same (disabled) Re-ID features, and the same association hyperparameters. Only the motion model and compensation module differ.

## Key Hyperparameters

All hyperparameters are in `configs/defaults.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `kalman.cv.std_weight_position` | 1e-2 | CV position process noise (scaled by bbox height) |
| `kalman.cv.std_weight_velocity` | 1e-5 | CV velocity process noise |
| `kalman.ca.std_weight_acceleration` | 2.5e-2 | CA acceleration process noise (the key tuning knob) |
| `tracker.max_cosine_distance` | 0.3 | Re-ID cosine distance gate |
| `tracker.max_iou_distance` | 0.7 | IoU fallback gate |
| `tracker.nn_budget` | 100 | Max gallery size per track |
| `tracker.max_age` | 30 | Frames before track deletion |
| `tracker.n_init` | 3 | Hits before track confirmation |
| `camera_motion.num_iterations` | 50 | ECC max iterations |
| `camera_motion.termination_eps` | 1e-3 | ECC convergence threshold |
| `camera_motion.downscale_factor` | 2 | Process at half resolution |

## Evaluation Metrics

| Metric | Description | Source |
|--------|-------------|--------|
| **MOTA** | Multi-Object Tracking Accuracy | `evaluation/mot_metrics.py` |
| **IDF1** | ID F1 Score (primary metric for identity preservation) | `evaluation/mot_metrics.py` |
| **IDS** | Identity Switches (lower is better) | `evaluation/mot_metrics.py` |
| **FP / FN** | False Positives / False Negatives | `evaluation/mot_metrics.py` |
| **MT / ML** | Mostly Tracked / Mostly Lost | `evaluation/mot_metrics.py` |
| **Frag** | Track Fragmentations | `evaluation/mot_metrics.py` |
| **Regime breakdown** | IDS by cause (accel / pan / both / neither) | `evaluation/regime_analysis.py` |
| **Prediction error** | Center displacement (px) at horizons 1/5/10/15/20 | `scripts/evaluate_prediction.py` |

## Quick Start (Minimal Replication)

If you just want to verify the pipeline works on a small subset, point `--output_dir` at a **local** folder (for example `results/local_val_run`); only the default paths `results/soccer_eval` and `results/soccernet_eval` match the hardcoded inputs to `generate_results_figures.py`.

```bash
# Activate environment
source .venv/bin/activate

# Run ablation on val only (writes under your chosen directory; not committed)
python scripts/run_all.py \
    --data_root soccer \
    --output_dir results/local_val_run \
    --splits val \
    --skip_detections

# Regenerate figures from the committed full-run JSON (after Steps 1–3 on default paths)
python scripts/generate_results_figures.py
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'numpy'` | Make sure you activated the venv: `source .venv/bin/activate` |
| ECC convergence warnings | Normal for shot changes; the code falls back to identity warp |
| `det/det.txt` missing | Run `python tools/generate_detections.py --data_root <path>` |
| CUDA out of memory | Use a smaller YOLO model: `--yolo_model yolov8m.pt` |
