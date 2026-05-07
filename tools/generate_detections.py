"""
Generate MOTChallenge-format detections for all sequences using YOLOv8.

Usage:
    python tools/generate_detections.py \
        --data_root SportsMOT_example/dataset/train \
        --model yolov8x.pt \
        --conf 0.3

Writes det/det.txt into each sequence folder, in the format:
    frame, -1, x, y, w, h, conf, -1, -1, -1
"""

import argparse
import os

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


PERSON_CLASS_ID = 0  # COCO class 0 = person


def generate_dets_for_sequence(model, seq_dir, conf_threshold=0.3, imgsz=1280):
    img_dir = os.path.join(seq_dir, "img1")
    if not os.path.isdir(img_dir):
        return

    image_files = sorted(
        f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))
    )
    if not image_files:
        return

    det_dir = os.path.join(seq_dir, "det")
    os.makedirs(det_dir, exist_ok=True)
    det_file = os.path.join(det_dir, "det.txt")

    lines = []
    for frame_idx, fname in enumerate(tqdm(image_files, desc=os.path.basename(seq_dir)), start=1):
        img_path = os.path.join(img_dir, fname)
        results = model(img_path, conf=conf_threshold, imgsz=imgsz, verbose=False)

        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                if cls != PERSON_CLASS_ID:
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                w = x2 - x1
                h = y2 - y1
                lines.append(f"{frame_idx},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.6f},-1,-1,-1\n")

    with open(det_file, "w") as f:
        f.writelines(lines)

    print(f"  Wrote {len(lines)} detections to {det_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate YOLOv8 detections in MOTChallenge format")
    parser.add_argument("--data_root", required=True, help="Directory whose subfolders are sequences")
    parser.add_argument("--model", default="yolov8x.pt", help="YOLOv8 model name or path")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    sequences = sorted(
        d for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d, "img1"))
    )

    print(f"Found {len(sequences)} sequences in {args.data_root}")

    for seq_name in sequences:
        seq_dir = os.path.join(args.data_root, seq_name)
        generate_dets_for_sequence(model, seq_dir, conf_threshold=args.conf, imgsz=args.imgsz)

    print("Done.")


if __name__ == "__main__":
    main()
