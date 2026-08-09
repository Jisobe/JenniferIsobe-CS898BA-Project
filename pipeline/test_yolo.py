"""
RollCall - YOLO Pipeline Evaluation

Evaluates the fine-tuned YOLO model on the held-out test split
(data/labeled/test/) via the exact same code path app.py uses --
DiceInferencePipeline.process_photo() -- so these numbers describe what a
real user actually sees, not an idealized measurement.

This replaces test_interface.py's two-stage "localization tax" comparison.
There's no separate Stage 1 (oracle-crop classifier) number here, because
there's no standalone classifier anymore -- YOLO localizes and classifies
in one pass, so there's nothing to compare it against in isolation.
test_locate_dice.py and test_model.py/test.py are similarly retired; their
job (measuring localization and classification in isolation) doesn't apply
to a single-stage detector.

What this script still measures, and why each number matters:

  Detection recall:
      Fraction of ground-truth dice that got ANY matching predicted box
      (IoU >= --min-match-iou). Analogous to the old pipeline's detection
      recall from locate_dice(), now measuring YOLO's box head instead of
      the classical CV ensemble.

  Hallucination rate:
      Fraction of predicted boxes that didn't match any ground-truth die.
      Same definition as before -- still the other half of the story
      recall alone doesn't tell you.

  Accuracy among matched dice:
      Of the dice YOLO actually found, how often was the face value
      correct? Comparable in spirit to the old "Stage 1 accuracy" (how
      good is classification given a correct-ish crop), but on YOLO's own
      real boxes rather than ground-truth crops, since there's no separate
      oracle-crop step to measure anymore.

  End-to-end accuracy:
      Correct-face predictions / ALL ground-truth dice. A missed die
      counts as wrong here (denominator is every ground-truth die, not
      just matched ones) -- a real player gets no digital die shown for
      one YOLO never found, same accounting rule test_interface.py used.

Directly comparable to the retired pipeline's historical numbers (~93%
oracle-crop / ~52% end-to-end, driven by ~63% recall / ~25% hallucination)
if you want a single-stage vs. two-stage comparison slide.

Usage:
    uv run test_yolo.py --weights runs/yolo/run_01/weights/best.pt
    uv run test_yolo.py --weights runs/yolo/run_01/weights/best.pt --min-match-iou 0.5

Outputs (default: <run dir>/test_results/):
    metrics.json                - all numbers, machine-readable
    confusion_matrix.png        - face-value confusion matrix (matched dice only)
    classification_report.txt   - per-class precision/recall/F1
    per_image_details.json      - per-photo breakdown for debugging specific failures
"""

import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from interface import DiceInferencePipeline

NUM_CLASSES = 6
CLASS_NAMES = ["1", "2", "3", "4", "5", "6"]

# A detected box must overlap a ground-truth box by at least this much IoU
# to be considered "the same die" rather than a coincidental overlap.
MIN_MATCH_IOU = 0.3


def box_iou(box1, box2):
    """IoU of two (x, y, w, h) boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union if union > 0 else 0.0


def yolo_to_pixel(cx, cy, bw, bh, img_w, img_h):
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return x1, y1, x2, y2


def load_ground_truth(labels_dir: Path, img_stem: str, img_w: int, img_h: int):
    """Parse a YOLO-format label file into pixel-space boxes + 1-6 face
    labels. Assumes the class index in the label file (0-5) maps to face
    value (index + 1) in the same order as data.yaml's `names` list --
    verify this holds for your export before trusting these numbers."""
    label_path = labels_dir / f"{img_stem}.txt"
    boxes = []
    if not label_path.exists():
        return boxes

    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_index = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
        except ValueError:
            continue
        if class_index < 0 or class_index > 5:
            continue

        x1, y1, x2, y2 = yolo_to_pixel(cx, cy, bw, bh, img_w, img_h)
        boxes.append({"box": (x1, y1, x2 - x1, y2 - y1), "label": class_index + 1})

    return boxes


def match_boxes(gt_boxes, pred_boxes, min_iou=MIN_MATCH_IOU):
    """Greedy one-to-one IoU matching: consider every (gt, pred) pair above
    the IoU threshold, assign highest-IoU pairs first, each box used at
    most once. Returns list of (gt_index, pred_index, iou)."""
    candidates = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred_box in enumerate(pred_boxes):
            iou = box_iou(gt["box"], pred_box)
            if iou >= min_iou:
                candidates.append((iou, gi, pi))
    candidates.sort(key=lambda c: -c[0])

    matched_gt, matched_pred = set(), set()
    matches = []
    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi, iou))

    return matches, matched_gt, matched_pred


def confusion_matrix(labels, preds, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, preds):
        cm[true_label, pred_label] += 1
    return cm


def per_class_prf1(cm):
    """Precision/recall/F1 per class computed directly from the confusion
    matrix (rows = true label, cols = predicted label)."""
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    support = cm.sum(axis=1)

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = (
            2 * precision[c] * recall[c] / (precision[c] + recall[c])
            if (precision[c] + recall[c]) > 0
            else 0.0
        )

    return precision, recall, f1, support


def plot_confusion_matrix(cm, class_names, out_path):
    cm_normalized = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, data, title, fmt in (
        (ax1, cm, "Confusion Matrix (counts)", "d"),
        (ax2, cm_normalized, "Confusion Matrix (row-normalized)", ".2f"),
    ):
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted face")
        ax.set_ylabel("True face")
        ax.set_title(title)

        thresh = data.max() / 2.0 if data.max() > 0 else 0.5
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                ax.text(
                    j, i, f"{value:{fmt}}",
                    ha="center", va="center",
                    color="white" if value > thresh else "black",
                    fontsize=9,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_classification_report(class_names, precision, recall, f1, support, out_path):
    lines = []
    header = f"{'class':<8}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, name in enumerate(class_names):
        lines.append(
            f"{name:<8}{precision[i]:>10.3f}{recall[i]:>10.3f}{f1[i]:>10.3f}{support[i]:>10d}"
        )
    lines.append("-" * len(header))
    macro_p, macro_r, macro_f1 = precision.mean(), recall.mean(), f1.mean()
    total_support = int(support.sum())
    lines.append(
        f"{'macro avg':<8}{macro_p:>10.3f}{macro_r:>10.3f}{macro_f1:>10.3f}{total_support:>10d}"
    )
    report = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(report + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="RollCall YOLO pipeline evaluation: end-to-end accuracy on the held-out test split"
    )
    parser.add_argument(
        "--weights", type=Path, required=True,
        help="Path to fine-tuned YOLO weights (e.g. runs/yolo/run_01/weights/best.pt)",
    )
    parser.add_argument(
        "--images-dir", type=Path, default=Path("data/labeled/test/images"),
        help="Raw full-photo images with YOLO labels, default: data/labeled/test/images",
    )
    parser.add_argument(
        "--labels-dir", type=Path, default=None,
        help="YOLO label directory (default: <images-dir>/../labels)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Where to save results (default: <weights run dir>/test_results)",
    )
    parser.add_argument(
        "--min-match-iou", type=float, default=MIN_MATCH_IOU,
        help=f"Minimum IoU to count a detection as matching a ground-truth die (default: {MIN_MATCH_IOU})",
    )
    args = parser.parse_args()

    labels_dir = args.labels_dir or (args.images_dir.parent / "labels")
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # weights live at <run_dir>/weights/best.pt -- default output alongside the run
    output_dir = args.output or (args.weights.parent.parent / "test_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in args.images_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not image_paths:
        raise RuntimeError(f"No images found under {args.images_dir}")
    print(f"Evaluating {len(image_paths)} images from {args.images_dir}")

    pipeline = DiceInferencePipeline(args.weights)

    matched_labels, matched_preds = [], []
    total_gt_dice = 0
    total_matched_gt = 0
    total_pred_boxes = 0
    total_matched_pred = 0
    matched_ious = []
    per_image_details = []
    skipped_no_labels = 0

    for img_path in image_paths:
        image_bgr = cv.imread(str(img_path))
        if image_bgr is None:
            print(f"  Could not read image, skipping: {img_path.name}")
            continue
        img_h, img_w = image_bgr.shape[:2]

        gt_boxes = load_ground_truth(labels_dir, img_path.stem, img_w, img_h)
        if not gt_boxes:
            skipped_no_labels += 1
            continue

        expected_count = len(gt_boxes)
        total_gt_dice += expected_count

        detections = pipeline.process_photo(image_bgr, expected_dice_count=expected_count)
        pred_boxes = [det.box for det in detections]
        total_pred_boxes += len(pred_boxes)

        matches, matched_gt_idx, matched_pred_idx = match_boxes(
            gt_boxes, pred_boxes, min_iou=args.min_match_iou
        )
        total_matched_gt += len(matched_gt_idx)
        total_matched_pred += len(matched_pred_idx)

        image_correct = 0
        for gi, pi, iou in matches:
            matched_ious.append(iou)
            det = detections[pi]
            gt_label = gt_boxes[gi]["label"]
            matched_labels.append(gt_label - 1)
            matched_preds.append(det.predicted_face - 1)
            if det.predicted_face == gt_label:
                image_correct += 1

        missed = expected_count - len(matched_gt_idx)
        hallucinated = len(pred_boxes) - len(matched_pred_idx)

        per_image_details.append({
            "image": img_path.name,
            "expected_dice": expected_count,
            "detected_boxes": len(pred_boxes),
            "matched": len(matched_gt_idx),
            "missed": missed,
            "hallucinated": hallucinated,
            "correct_of_matched": image_correct,
        })

    if skipped_no_labels:
        print(f"Skipped {skipped_no_labels} image(s) with no ground-truth label file")

    # ---- Aggregate metrics ----
    matched_labels_arr = np.array(matched_labels)
    matched_preds_arr = np.array(matched_preds)
    correct_matched = int((matched_labels_arr == matched_preds_arr).sum())

    end_to_end_accuracy = correct_matched / total_gt_dice if total_gt_dice > 0 else None
    accuracy_among_matched = correct_matched / total_matched_gt if total_matched_gt > 0 else None
    detection_recall = total_matched_gt / total_gt_dice if total_gt_dice > 0 else None
    hallucination_rate = (
        (total_pred_boxes - total_matched_pred) / total_pred_boxes if total_pred_boxes > 0 else None
    )
    mean_matched_iou = float(np.mean(matched_ious)) if matched_ious else None

    print(f"\n{'=' * 60}")
    print("  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Ground-truth dice evaluated:      {total_gt_dice}")
    print(f"  Detection recall:                 {detection_recall:.4f}" if detection_recall is not None else "  Detection recall: N/A")
    print(f"  Hallucination rate:                {hallucination_rate:.4f}" if hallucination_rate is not None else "  Hallucination rate: N/A")
    print(f"  Mean IoU (matched boxes):          {mean_matched_iou:.4f}" if mean_matched_iou is not None else "  Mean matched IoU: N/A")
    print(f"  Accuracy among matched dice:       {accuracy_among_matched:.4f}" if accuracy_among_matched is not None else "  Accuracy among matched: N/A")
    print(f"  End-to-end accuracy (all GT dice): {end_to_end_accuracy:.4f}" if end_to_end_accuracy is not None else "  End-to-end accuracy: N/A")
    print(f"{'=' * 60}")

    if len(matched_labels_arr) > 0:
        cm = confusion_matrix(matched_labels_arr, matched_preds_arr, NUM_CLASSES)
        precision, recall, f1, support = per_class_prf1(cm)
        write_classification_report(
            CLASS_NAMES, precision, recall, f1, support, output_dir / "classification_report.txt"
        )
        plot_confusion_matrix(cm, CLASS_NAMES, output_dir / "confusion_matrix.png")

    metrics = {
        "weights": str(args.weights),
        "images_dir": str(args.images_dir),
        "num_images_evaluated": len(image_paths) - skipped_no_labels,
        "min_match_iou": args.min_match_iou,
        "total_ground_truth_dice": total_gt_dice,
        "detection_recall": detection_recall,
        "hallucination_rate": hallucination_rate,
        "mean_matched_iou": mean_matched_iou,
        "accuracy_among_matched_dice": accuracy_among_matched,
        "end_to_end_accuracy": end_to_end_accuracy,
        "total_matched_gt": total_matched_gt,
        "total_predicted_boxes": total_pred_boxes,
        "total_matched_predictions": total_matched_pred,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "per_image_details.json", "w") as f:
        json.dump(per_image_details, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()