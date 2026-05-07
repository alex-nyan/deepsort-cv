"""
Run tracker on a single MOTChallenge-format sequence.

Usage:
    python scripts/run_tracker.py \
        --config configs/defaults.yaml \
        --sequence /path/to/sequence \
        --output /path/to/output.txt \
        --kalman_model constant_acceleration \
        --camera_motion_compensation

Sequence directory expected structure (MOTChallenge format):
    sequence/
    ├── det/
    │   └── det.txt            # Detections: frame,id,x,y,w,h,conf,-1,-1,-1
    ├── gt/
    │   └── gt.txt             # Ground truth (for evaluation)
    └── img1/
        ├── 000001.jpg
        ├── 000002.jpg
        └── ...
"""

import argparse
import os
import sys

import cv2
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deep_sort.sort.tracker import Tracker
from deep_sort.sort.detection import Detection
from deep_sort.sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.sort.preprocessing import non_max_suppression
from deep_sort.reid.extractor import build_extractor
from camera_motion.ecc_compensator import ECCCameraCompensator


def load_detections(det_file):
    """
    Load detections from MOTChallenge format file.

    Returns
    -------
    dict: frame_id -> list of (tlwh, confidence)
    """
    detections = {}
    with open(det_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame = int(parts[0])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = float(parts[6])
            detections.setdefault(frame, []).append((np.array([x, y, w, h]), conf))
    return detections


def run_tracker(config, sequence_dir, output_file, kalman_model=None,
                camera_motion_compensation=None, reid_model_path=None):
    """
    Run tracking on a single sequence.

    Parameters
    ----------
    config : dict
        Loaded YAML config.
    sequence_dir : str
    output_file : str
    kalman_model : str | None
        Override config's kalman_model.
    camera_motion_compensation : bool | None
        Override config's camera_motion_compensation.
    reid_model_path : str | None
        Path to Re-ID model weights.
    """
    # --- Resolve configuration ---
    tracker_cfg = config["tracker"]
    km = kalman_model or tracker_cfg["kalman_model"]
    cmc_enabled = camera_motion_compensation if camera_motion_compensation is not None \
        else tracker_cfg["camera_motion_compensation"]

    # Kalman parameters
    if km == "constant_velocity":
        kf_params = config["kalman"]["cv"]
    else:
        kf_params = config["kalman"]["ca"]

    # --- Build components ---
    metric = NearestNeighborDistanceMetric(
        "cosine",
        matching_threshold=tracker_cfg["max_cosine_distance"],
        budget=tracker_cfg["nn_budget"],
    )

    compensator = None
    if cmc_enabled:
        cmc_cfg = config["camera_motion"]
        compensator = ECCCameraCompensator(
            warp_mode=cmc_cfg["warp_mode"],
            num_iterations=cmc_cfg["num_iterations"],
            termination_eps=cmc_cfg["termination_eps"],
            downscale_factor=cmc_cfg["downscale_factor"],
            gaussian_blur_sigma=cmc_cfg["gaussian_blur_sigma"],
        )

    tracker = Tracker(
        metric,
        kalman_model=km,
        camera_compensator=compensator,
        max_iou_distance=tracker_cfg["max_iou_distance"],
        max_age=tracker_cfg["max_age"],
        n_init=tracker_cfg["n_init"],
        kalman_params=kf_params,
    )

    # Feature extractor
    extractor = build_extractor(
        extractor_type="resnet" if reid_model_path else "dummy",
        model_path=reid_model_path,
    )

    # --- Load data ---
    det_file = os.path.join(sequence_dir, "det", "det.txt")
    img_dir = os.path.join(sequence_dir, "img1")
    raw_detections = load_detections(det_file)

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    # Determine frame range
    image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".png"))
    ])
    num_frames = len(image_files)

    # --- Track ---
    results = []
    warp_magnitudes = {}  # for regime analysis

    conf_threshold = config["detection"]["confidence_threshold"]
    nms_overlap = config["detection"]["nms_max_overlap"]

    for frame_idx in tqdm(range(1, num_frames + 1), desc="Tracking"):
        # Load frame
        img_path = os.path.join(img_dir, image_files[frame_idx - 1])
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        # Get detections for this frame
        frame_dets = raw_detections.get(frame_idx, [])

        # Filter by confidence
        frame_dets = [(tlwh, conf) for tlwh, conf in frame_dets if conf >= conf_threshold]

        if len(frame_dets) > 0:
            boxes = np.array([d[0] for d in frame_dets])
            scores = np.array([d[1] for d in frame_dets])

            # NMS
            indices = non_max_suppression(boxes, nms_overlap, scores)
            boxes = boxes[indices]
            scores = scores[indices]

            # Extract Re-ID features
            features = extractor(frame, boxes)

            # Create Detection objects
            detections = [
                Detection(box, score, feat)
                for box, score, feat in zip(boxes, scores, features)
            ]
        else:
            detections = []

        # Predict (with optional camera compensation)
        tracker.predict(frame=frame)

        # Record warp magnitude for regime analysis
        if compensator is not None:
            warp_magnitudes[frame_idx] = compensator.last_translation_magnitude

        # Update
        tracker.update(detections)

        # Record results in MOTChallenge format
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append(
                f"{frame_idx},{track.track_id},{bbox[0]:.2f},{bbox[1]:.2f},"
                f"{bbox[2]:.2f},{bbox[3]:.2f},1,-1,-1,-1\n"
            )

    # --- Write output ---
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        f.writelines(results)

    print(f"Wrote {len(results)} detections to {output_file}")

    return warp_magnitudes


def main():
    parser = argparse.ArgumentParser(description="Run DeepSORT tracker")
    parser.add_argument("--config", default="configs/defaults.yaml")
    parser.add_argument("--sequence", required=True, help="Path to sequence directory")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--kalman_model", choices=["constant_velocity", "constant_acceleration"])
    parser.add_argument("--camera_motion_compensation", action="store_true", default=None)
    parser.add_argument("--no_camera_motion_compensation", dest="camera_motion_compensation",
                        action="store_false")
    parser.add_argument("--reid_model", default=None, help="Path to Re-ID model weights")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_tracker(
        config, args.sequence, args.output,
        kalman_model=args.kalman_model,
        camera_motion_compensation=args.camera_motion_compensation,
        reid_model_path=args.reid_model,
    )


if __name__ == "__main__":
    main()
