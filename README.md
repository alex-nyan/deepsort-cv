# Acceleration-Aware DeepSORT with Camera-Motion Compensation

Multi-object tracking for broadcast soccer, extending DeepSORT with:
1. **Constant-acceleration Kalman filter** (12D state vector) for bursty player dynamics
2. **ECC camera-motion compensation** to correct for broadcast pan/zoom

## Project Structure

```
accel-deepsort/
├── deep_sort/
│   ├── sort/
│   │   ├── kalman_filter.py        # Original 8D constant-velocity KF
│   │   ├── kalman_filter_accel.py  # 12D constant-acceleration KF
│   │   ├── track.py                # Single-target track state machine
│   │   ├── tracker.py              # Multi-target tracker (predict→associate loop)
│   │   ├── nn_matching.py          # Nearest-neighbor matching (cosine/euclidean)
│   │   ├── linear_assignment.py    # Hungarian + cascade matching
│   │   ├── iou_matching.py         # IoU-based fallback matching
│   │   ├── detection.py            # Detection data class
│   │   └── preprocessing.py        # NMS and utilities
│   └── reid/
│       └── extractor.py            # Re-ID feature extraction
├── camera_motion/
│   └── ecc_compensator.py          # ECC affine registration module
├── evaluation/
│   ├── mot_metrics.py              # MOTA, IDF1, ID-switch computation
│   └── regime_analysis.py          # Per-regime breakdown (accel events, pan events)
├── configs/
│   └── defaults.yaml               # All hyperparameters
├── scripts/
│   ├── run_all.py                  # Single command to run everything
│   ├── run_tracker.py              # Main entry point
│   ├── run_ablation.py             # Four-way ablation driver (with regime analysis)
│   ├── run_botsort_baseline.py     # YOLOv8 + BoT-SORT external baseline
│   ├── run_full_evaluation.py      # Complete evaluation pipeline
│   └── evaluate_prediction.py      # KF prediction accuracy vs true future
├── tools/
│   ├── generate_detections.py      # YOLOv8 detection generation
│   ├── convert_soccernet.py        # SoccerNet → MOTChallenge format
│   └── convert_sportsmot.py        # SportsMOT → MOTChallenge format
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Datasets

- **SoccerNet-Tracking**: https://www.soccer-net.org/ (requires registration)
- **SportsMOT**: https://github.com/MCG-NJU/SportsMOT

Both must be converted to MOTChallenge format using scripts in `tools/`.

### Data Preparation

**Option A — SportsMOT (used in our experiments):**

1. Download SportsMOT from the link above and place it under `sportsmot_publish/`.
2. Convert the soccer sequences to MOTChallenge layout:

```bash
python tools/convert_sportsmot.py \
    --sportsmot_root sportsmot_publish/dataset \
    --output_dir soccer \
    --sport soccer
```

This creates `soccer/train/`, `soccer/val/`, `soccer/test/` with symlinks to the original data.

**Option B — SoccerNet-Tracking:**

```bash
python tools/convert_soccernet.py \
    --soccernet_root /path/to/soccernet \
    --output_dir soccernet_data
```

### Generating Detections

Sequences require a `det/det.txt` file for tracking. Generate detections with YOLOv8:

```bash
python tools/generate_detections.py --data_root soccer/train --model yolov8x.pt
python tools/generate_detections.py --data_root soccer/val   --model yolov8x.pt
```

## Reproducing Results

### Run Everything (Recommended)

A single script runs the full pipeline — detection generation, four-way ablation, prediction evaluation, and BoT-SORT baseline:

```bash
python scripts/run_all.py
```

This defaults to `soccer/` as the data root and evaluates on `train` + `val` splits. Results are saved to `results/full_eval/`.

**Useful flags:**

```bash
# Skip detection generation (if det.txt files already exist)
python scripts/run_all.py --skip_detections

# Skip individual stages
python scripts/run_all.py --skip_botsort
python scripts/run_all.py --skip_prediction

# Use a different YOLO model
python scripts/run_all.py --yolo_model yolov8m.pt

# Evaluate only on the val split
python scripts/run_all.py --splits val

# Full customization
python scripts/run_all.py \
    --config configs/defaults.yaml \
    --data_root soccer \
    --output_dir results/full_eval \
    --splits train val \
    --yolo_model yolov8x.pt
```

### Run Individual Evaluations

**1. Four-way ablation** (MOTA, IDF1, ID switches + regime analysis):

```bash
python scripts/run_ablation.py \
    --config configs/defaults.yaml \
    --data_root soccer/train \
    --output_dir results/ablation
```

**2. Kalman filter prediction accuracy** (CV 8D vs CA 12D, pixel error at 1/5/10/15/20 frames):

```bash
python scripts/evaluate_prediction.py --gt_file soccer/train/v_1yHWGw8DH4A_c029/gt/gt.txt
```

**3. BoT-SORT baseline** (YOLOv8 + BoT-SORT for external comparison):

```bash
python scripts/run_botsort_baseline.py --data_root soccer/train --model yolov8x.pt
```

**4. Single sequence tracking:**

```bash
python scripts/run_tracker.py \
    --config configs/defaults.yaml \
    --sequence soccer/train/v_1yHWGw8DH4A_c029 \
    --output results/track.txt \
    --kalman_model constant_acceleration \
    --camera_motion_compensation
```

## Evaluation Metrics

| Metric | Description | Source |
|--------|-------------|--------|
| **MOTA** | Multi-Object Tracking Accuracy | `evaluation/mot_metrics.py` |
| **IDF1** | ID F1 Score (primary metric) | `evaluation/mot_metrics.py` |
| **IDS** | Identity Switches | `evaluation/mot_metrics.py` |
| **FP / FN** | False Positives / False Negatives | `evaluation/mot_metrics.py` |
| **MT / ML** | Mostly Tracked / Mostly Lost | `evaluation/mot_metrics.py` |
| **Frag** | Track Fragmentations | `evaluation/mot_metrics.py` |
| **Regime breakdown** | ID switches by cause (accel / pan / both / neither) | `evaluation/regime_analysis.py` |
| **Prediction error** | Displacement error (px) at multiple horizons | `scripts/evaluate_prediction.py` |

## Ablation Configurations

| Config | Accel KF | Camera Comp | Description |
|--------|----------|-------------|-------------|
| baseline | ✗ | ✗ | Original DeepSORT |
| accel_only | ✓ | ✗ | 12D KF, no camera comp |
| cmc_only | ✗ | ✓ | 8D KF + ECC compensation |
| full | ✓ | ✓ | Both modifications |

## Output

All results are saved as JSON in the output directory:

| File | Contents |
|------|----------|
| `ablation_results.json` | Per-sequence MOTA, IDF1, IDS, FP, FN + regime breakdown |
| `prediction_results.json` | CV vs CA prediction error at each horizon |
| `botsort_results.json` | BoT-SORT per-sequence metrics |
| `all_results.json` | Combined results from all evaluations |
