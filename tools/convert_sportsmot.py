"""
Convert SportsMOT dataset to MOTChallenge format.

SportsMOT is already close to MOTChallenge format, but may need
minor adjustments (path structure, annotation format).

Usage:
    python tools/convert_sportsmot.py \
        --sportsmot_root /path/to/sportsmot \
        --output_dir /path/to/output \
        --sport soccer
"""

import argparse
import os
import shutil


def convert_sportsmot(sportsmot_root, output_dir, sport_filter=None):
    """
    Convert SportsMOT to clean MOTChallenge format.

    SportsMOT organizes data by sport (basketball, volleyball, soccer).
    We filter for soccer sequences.
    """
    # SportsMOT typically has splits: train, val, test
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(sportsmot_root, split)
        if not os.path.exists(split_dir):
            continue

        sequences = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])

        for seq_name in sequences:
            seq_dir = os.path.join(split_dir, seq_name)

            # Filter by sport if specified
            if sport_filter:
                # SportsMOT naming convention: v_<sport>_<id>
                sport_in_name = seq_name.lower()
                if sport_filter.lower() not in sport_in_name:
                    continue

            print(f"Converting {split}/{seq_name}...")

            output_seq_dir = os.path.join(output_dir, f"{split}_{seq_name}")
            os.makedirs(output_seq_dir, exist_ok=True)

            # Copy/symlink standard MOTChallenge structure
            for subdir in ["gt", "det", "img1"]:
                src = os.path.join(seq_dir, subdir)
                dst = os.path.join(output_seq_dir, subdir)

                if os.path.exists(src):
                    if os.path.islink(dst) or os.path.exists(dst):
                        continue
                    os.symlink(os.path.abspath(src), dst)

            # Some SportsMOT versions use different gt filenames
            gt_dir = os.path.join(output_seq_dir, "gt")
            if os.path.exists(gt_dir):
                gt_file = os.path.join(gt_dir, "gt.txt")
                if not os.path.exists(gt_file):
                    # Check for alternative names
                    for alt in ["gt_val.txt", "gt_train.txt"]:
                        alt_path = os.path.join(gt_dir, alt)
                        if os.path.exists(alt_path):
                            os.symlink(os.path.abspath(alt_path), gt_file)
                            break

    print(f"Done. Output at {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert SportsMOT to MOTChallenge format")
    parser.add_argument("--sportsmot_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sport", default=None, help="Filter by sport (e.g., 'soccer')")

    args = parser.parse_args()
    convert_sportsmot(args.sportsmot_root, args.output_dir, args.sport)


if __name__ == "__main__":
    main()
