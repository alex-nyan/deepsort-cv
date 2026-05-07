"""
Run YOLOv8 + BoT-SORT as an external baseline for comparison.

This provides the reference point promised in the paper: a top-tier
modern tracker (BoT-SORT) using the same detector (YOLOv8) on the
same sequences. This lets us see how our DeepSORT modifications
compare to a state-of-the-art system.

Usage:
    python scripts/run_botsort_baseline.py \
        --data_root /path/to/dataset \
        --output_dir results/botsort_baseline \
        --model yolov8x.pt \
        --conf 0.3

Output is written in MOTChallenge format for evaluation with mot_metrics.py.
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ultralytics import YOLO


PERSON_CLASS_ID = 0


def run_botsort_on_sequence(model, seq_dir, output_file, conf=0.3, imgsz=1280):
    """
    Run YOLOv8 + BoT-SORT on a single MOTChallenge-format sequence.

    Parameters
    ----------
    model : YOLO
        Loaded YOLO model.
    seq_dir : str
        Sequence directory with img1/ subfolder.
    output_file : str
        Output path for MOTChallenge-format results.
    conf : float
        Detection confidence threshold.
    imgsz : int
        Inference image size.
    """
    img_dir = os.path.join(seq_dir, "img1")
    if not os.path.isdir(img_dir):
        print(f"  Skipping: no img1/ directory in {seq_dir}")
        return

    image_files = sorted(
        f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))
    )
    if not image_files:
        print(f"  Skipping: no images found in {img_dir}")
        return

    results_lines = []

    for frame_idx, fname in enumerate(tqdm(image_files, desc=os.path.basename(seq_dir)), start=1):
        img_path = os.path.join(img_dir, fname)

        # Run YOLOv8 with BoT-SORT tracking
        results = model.track(
            img_path,
            conf=conf,
            imgsz=imgsz,
            persist=True,
            tracker="botsort.yaml",
            verbose=False,
            classes=[PERSON_CLASS_ID],
        )

        for r in results:
            boxes = r.boxes
            if boxes.id is None:
                continue

            for i in range(len(boxes)):
                track_id = int(boxes.id[i])
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf_score = float(boxes.conf[i])
                w = x2 - x1
                h = y2 - y1
                results_lines.append(
                    f"{frame_idx},{track_id},{x1:.2f},{y1:.2f},"
                    f"{w:.2f},{h:.2f},{conf_score:.4f},-1,-1,-1\n"
                )

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        f.writelines(results_lines)

    print(f"  Wrote {len(results_lines)} tracked detections to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 + BoT-SORT baseline on MOTChallenge sequences"
    )
    parser.add_argument("--data_root", required=True, help="Dataset root with sequence subdirs")
    parser.add_argument("--output_dir", default="results/botsort_baseline")
    parser.add_argument("--model", default="yolov8x.pt", help="YOLOv8 model")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    sequences = sorted(
        d for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d, "img1"))
    )

    if not sequences:
        print(f"No sequences found in {args.data_root}")
        return

    print(f"Found {len(sequences)} sequences")
    os.makedirs(args.output_dir, exist_ok=True)

    for seq_name in sequences:
        seq_dir = os.path.join(args.data_root, seq_name)
        output_file = os.path.join(args.output_dir, f"{seq_name}.txt")
        print(f"\nProcessing: {seq_name}")

        # Reset tracker state between sequences
        model.predictor = None
        run_botsort_on_sequence(model, seq_dir, output_file, conf=args.conf, imgsz=args.imgsz)

    # Evaluate if ground truth is available
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    from evaluation.mot_metrics import evaluate_sequence

    header = f"{'Sequence':<40}{'MOTA':>10}{'IDF1':>10}{'IDS':>8}{'FP':>8}{'FN':>8}"
    print(header)
    print("-" * len(header))

    all_metrics = []
    for seq_name in sequences:
        gt_file = os.path.join(args.data_root, seq_name, "gt", "gt.txt")
        result_file = os.path.join(args.output_dir, f"{seq_name}.txt")

        if not os.path.exists(gt_file):
            continue

        try:
            metrics = evaluate_sequence(gt_file, result_file)
            all_metrics.append(metrics)
            print(f"{seq_name:<40}{metrics['MOTA']:>10.4f}{metrics['IDF1']:>10.4f}"
                  f"{metrics['IDS']:>8}{metrics['FP']:>8}{metrics['FN']:>8}")
        except Exception as e:
            print(f"{seq_name:<40} Error: {e}")

    if all_metrics:
        print("-" * len(header))
        avg_mota = np.mean([m["MOTA"] for m in all_metrics])
        avg_idf1 = np.mean([m["IDF1"] for m in all_metrics])
        total_ids = sum(m["IDS"] for m in all_metrics)
        total_fp = sum(m["FP"] for m in all_metrics)
        total_fn = sum(m["FN"] for m in all_metrics)
        print(f"{'AVERAGE/TOTAL':<40}{avg_mota:>10.4f}{avg_idf1:>10.4f}"
              f"{total_ids:>8}{total_fp:>8}{total_fn:>8}")


if __name__ == "__main__":
    main()
