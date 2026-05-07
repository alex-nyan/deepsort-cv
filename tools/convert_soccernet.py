"""
Convert SoccerNet-Tracking annotations to MOTChallenge format.

SoccerNet-Tracking uses a JSON-based annotation format.
This script converts it to the standard MOTChallenge txt format
expected by our tracker and evaluation code.

Usage:
    python tools/convert_soccernet.py \
        --soccernet_root /path/to/soccernet \
        --output_dir /path/to/output
"""

import argparse
import json
import os


def convert_sequence(annotation_file, image_dir, output_dir):
    """
    Convert a single SoccerNet-Tracking sequence.

    Parameters
    ----------
    annotation_file : str
        Path to SoccerNet JSON annotation file.
    image_dir : str
        Path to directory containing frame images.
    output_dir : str
        Output directory in MOTChallenge format.
    """
    with open(annotation_file, "r") as f:
        data = json.load(f)

    # Create MOTChallenge directory structure
    gt_dir = os.path.join(output_dir, "gt")
    det_dir = os.path.join(output_dir, "det")
    img_dir = os.path.join(output_dir, "img1")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(det_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    gt_lines = []
    det_lines = []

    # SoccerNet format varies by version — handle common structures
    if "images" in data:
        # SoccerNet v2/v3 COCO-like format
        image_id_to_frame = {}
        for img_info in data["images"]:
            image_id_to_frame[img_info["id"]] = img_info.get("frame_id", img_info["id"])
            # Symlink or copy images
            src = os.path.join(image_dir, img_info["file_name"])
            if os.path.exists(src):
                frame_num = image_id_to_frame[img_info["id"]]
                dst = os.path.join(img_dir, f"{frame_num:06d}.jpg")
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(src), dst)

        for ann in data.get("annotations", []):
            frame = image_id_to_frame.get(ann["image_id"], ann["image_id"])
            track_id = ann.get("track_id", ann.get("id", 0))
            bbox = ann["bbox"]  # [x, y, w, h]

            # Ground truth line
            gt_lines.append(
                f"{frame},{track_id},{bbox[0]:.2f},{bbox[1]:.2f},"
                f"{bbox[2]:.2f},{bbox[3]:.2f},1,1,1.0\n"
            )

            # Also use as detections (or you can use a separate detector)
            conf = ann.get("score", 1.0)
            det_lines.append(
                f"{frame},-1,{bbox[0]:.2f},{bbox[1]:.2f},"
                f"{bbox[2]:.2f},{bbox[3]:.2f},{conf:.4f},-1,-1,-1\n"
            )
    else:
        print(f"Warning: Unrecognized SoccerNet format in {annotation_file}")
        return False

    # Write files
    with open(os.path.join(gt_dir, "gt.txt"), "w") as f:
        f.writelines(sorted(gt_lines))

    with open(os.path.join(det_dir, "det.txt"), "w") as f:
        f.writelines(sorted(det_lines))

    print(f"  Converted: {len(gt_lines)} GT annotations, {len(det_lines)} detections")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert SoccerNet to MOTChallenge format")
    parser.add_argument("--soccernet_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "test", "challenge"])

    args = parser.parse_args()

    split_dir = os.path.join(args.soccernet_root, args.split)
    if not os.path.exists(split_dir):
        # Try flat structure
        split_dir = args.soccernet_root

    sequences = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])

    print(f"Found {len(sequences)} sequences in {split_dir}")

    for seq_name in sequences:
        seq_dir = os.path.join(split_dir, seq_name)
        print(f"\nConverting {seq_name}...")

        # Look for annotation file
        ann_file = None
        for candidate in ["labels.json", "annotations.json", f"{seq_name}.json"]:
            path = os.path.join(seq_dir, candidate)
            if os.path.exists(path):
                ann_file = path
                break

        if ann_file is None:
            print(f"  Skipping: no annotation file found")
            continue

        output_seq_dir = os.path.join(args.output_dir, seq_name)
        convert_sequence(ann_file, seq_dir, output_seq_dir)

    print(f"\nDone. Output at {args.output_dir}")


if __name__ == "__main__":
    main()
