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

## Usage

```bash
# Run single configuration
python scripts/run_tracker.py --config configs/defaults.yaml --sequence <path> --output <path>

# Run four-way ablation
python scripts/run_ablation.py --config configs/defaults.yaml --data_root <path>

# Run YOLOv8 + BoT-SORT baseline for comparison
python scripts/run_botsort_baseline.py --data_root <path> --model yolov8x.pt

# Evaluate prediction accuracy (CV vs CA Kalman filter)
python scripts/evaluate_prediction.py --gt_file <path/to/gt.txt>

# Run full evaluation pipeline (ablation + BoT-SORT + prediction)
python scripts/run_full_evaluation.py --config configs/defaults.yaml --data_root <path>

# Generate detections with YOLOv8
python tools/generate_detections.py --data_root <path> --model yolov8x.pt
```

## Ablation Configurations

| Config | Accel KF | Camera Comp | Description |
|--------|----------|-------------|-------------|
| baseline | ✗ | ✗ | Original DeepSORT |
| accel_only | ✓ | ✗ | 12D KF, no camera comp |
| cmc_only | ✗ | ✓ | 8D KF + ECC compensation |
| full | ✓ | ✓ | Both modifications |
# deepsort-cv
