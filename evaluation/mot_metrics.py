"""
MOT evaluation metrics using the py-motmetrics library.

Primary metrics for the paper:
    - IDF1: Identity F1 score (primary — measures identity preservation)
    - MOTA: Multi-Object Tracking Accuracy
    - IDS: Number of identity switches
    - FP, FN: False positives, false negatives
"""

import os
import numpy as np

try:
    import motmetrics as mm

    MOTMETRICS_AVAILABLE = True
except ImportError:
    MOTMETRICS_AVAILABLE = False


def load_mot_results(filepath):
    """
    Load tracking results in MOTChallenge format.

    Format: frame, id, x, y, w, h, conf, -1, -1, -1

    Returns
    -------
    dict: frame_id -> list of (track_id, x, y, w, h)
    """
    results = {}
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            results.setdefault(frame, []).append((tid, x, y, w, h))
    return results


def load_mot_gt(filepath):
    """
    Load ground truth in MOTChallenge format.

    Format: frame, id, x, y, w, h, flag, class, visibility

    Returns
    -------
    dict: frame_id -> list of (track_id, x, y, w, h)
    """
    gt = {}
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

            # MOTChallenge: column 7 is "consider" flag (0=ignore, 1=consider)
            if len(parts) >= 7:
                flag = int(parts[6])
                if flag == 0:
                    continue

            gt.setdefault(frame, []).append((tid, x, y, w, h))
    return gt


def evaluate_sequence(gt_file, result_file, iou_threshold=0.5):
    """
    Evaluate a single sequence.

    Parameters
    ----------
    gt_file : str
        Path to ground truth file.
    result_file : str
        Path to tracker output file.
    iou_threshold : float
        IoU threshold for matching.

    Returns
    -------
    summary : dict
        Dictionary with MOTA, IDF1, IDS, FP, FN, etc.
    """
    if not MOTMETRICS_AVAILABLE:
        raise ImportError("motmetrics package required. pip install motmetrics")

    gt = load_mot_gt(gt_file)
    results = load_mot_results(result_file)

    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(list(gt.keys()) + list(results.keys())))

    for frame in all_frames:
        gt_objs = gt.get(frame, [])
        res_objs = results.get(frame, [])

        gt_ids = [o[0] for o in gt_objs]
        res_ids = [o[0] for o in res_objs]

        gt_boxes = np.array([[o[1], o[2], o[1] + o[3], o[2] + o[4]] for o in gt_objs])  # tlbr
        res_boxes = np.array([[o[1], o[2], o[1] + o[3], o[2] + o[4]] for o in res_objs])

        if len(gt_boxes) > 0 and len(res_boxes) > 0:
            distances = mm.distances.iou_matrix(gt_boxes, res_boxes, max_iou=1 - iou_threshold)
        else:
            distances = np.empty((len(gt_boxes), len(res_boxes)))

        acc.update(gt_ids, res_ids, distances)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=[
        "mota", "idf1", "num_switches", "num_false_positives",
        "num_misses", "mostly_tracked", "mostly_lost", "num_fragmentations",
    ], name="sequence")

    return {
        "MOTA": float(summary["mota"].iloc[0]),
        "IDF1": float(summary["idf1"].iloc[0]),
        "IDS": int(summary["num_switches"].iloc[0]),
        "FP": int(summary["num_false_positives"].iloc[0]),
        "FN": int(summary["num_misses"].iloc[0]),
        "MT": int(summary["mostly_tracked"].iloc[0]),
        "ML": int(summary["mostly_lost"].iloc[0]),
        "Frag": int(summary["num_fragmentations"].iloc[0]),
    }


def evaluate_dataset(gt_dir, result_dir, sequences=None, iou_threshold=0.5):
    """
    Evaluate all sequences in a dataset.

    Parameters
    ----------
    gt_dir : str
        Directory containing gt/gt.txt for each sequence.
    result_dir : str
        Directory containing <sequence>.txt result files.
    sequences : list of str | None
        Sequence names. If None, auto-detect.

    Returns
    -------
    per_sequence : dict of dict
    overall : dict
        Aggregated metrics.
    """
    if sequences is None:
        sequences = sorted([
            d for d in os.listdir(gt_dir)
            if os.path.isdir(os.path.join(gt_dir, d))
        ])

    per_sequence = {}
    for seq in sequences:
        gt_file = os.path.join(gt_dir, seq, "gt", "gt.txt")
        result_file = os.path.join(result_dir, f"{seq}.txt")

        if not os.path.exists(gt_file) or not os.path.exists(result_file):
            print(f"Skipping {seq}: missing files")
            continue

        per_sequence[seq] = evaluate_sequence(gt_file, result_file, iou_threshold)

    # Aggregate
    if per_sequence:
        overall = {}
        for key in ["IDS", "FP", "FN", "MT", "ML", "Frag"]:
            overall[key] = sum(s[key] for s in per_sequence.values())
        # MOTA and IDF1 are averages (simplification — proper way uses accumulator merging)
        for key in ["MOTA", "IDF1"]:
            overall[key] = np.mean([s[key] for s in per_sequence.values()])
    else:
        overall = {}

    return per_sequence, overall
