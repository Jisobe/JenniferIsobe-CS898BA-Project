# stage_1 - CNN Only
# stage_2 - E2E
import argparse
import json
from pathlib import Path
import cv2 as cv
import numpy as np
import torch
from PIL import Image
from locator.crop_script import yolo_to_pixel, add_padding, apply_clahe, PADDING_PX
from locator.upgraded_locate_dice import box_iou
from pipeline.interface import DiceInferencePipeline
from classifier.test.test import (
    load_model,
    build_test_transform,
    confusion_matrix,
    per_class_performance,
    plot_confusion_matrices,
    write_classification_report,
    NUM_CLASSES,
    CLASS_NAMES,
)

MIN_MATCH_IOU = 0.3

def load_ground_truth(labels_dir, img_stem, img_w, img_h):
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

        x1, y1, x2, y2 = yolo_to_pixel((cx, cy, bw, bh), img_w, img_h)
        boxes.append({"box": (x1, y1, x2 - x1, y2 - y1), "label": class_index + 1})

    return boxes

def match_boxes(gt_boxes, pred_boxes, min_iou=MIN_MATCH_IOU):
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

def crop_ground_truth_box(image_bgr, box, padding=PADDING_PX):
    x, y, w, h = box
    img_h, img_w = image_bgr.shape[:2]
    x1, y1, x2, y2 = add_padding(x, y, x + w, y + h, padding, img_w, img_h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return apply_clahe(crop)

def stage1_classify(model, transform, device, crop_bgr):
    crop_rgb = cv.cvtColor(crop_bgr, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    pred_idx = int(probs.argmax())
    return pred_idx + 1

def main():
    parser = argparse.ArgumentParser(
        description="RollCall integration test: Stage 1 (classifier) vs Stage 2 (full pipeline) accuracy"
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to a trained checkpoint (e.g. runs/custom/run_20/best_model.pt)",
    )
    parser.add_argument(
        "--images-dir", type=Path, default=Path("data/labeled/test/images"),
        help="Raw full-photo images with YOLO labels (Roboflow export), default: data/labeled/test/images",
    )
    parser.add_argument(
        "--labels-dir", type=Path, default=None,
        help="YOLO label directory (default: <images-dir>/../labels)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Where to save results (default: <checkpoint's run dir>/integration_results)",
    )
    parser.add_argument(
        "--min-match-iou", type=float, default=MIN_MATCH_IOU,
        help=f"Minimum IoU to count a detection as matching a ground-truth die (default: {MIN_MATCH_IOU})",
    )
    parser.add_argument(
        "--pip-mode", choices=["fallback", "always"], default="fallback",
        help="'fallback' (default): pip-clustering only tops up a shortfall. 'always': "
             "pip-clustering runs unconditionally and votes alongside the silhouette "
             "ensemble -- targets touching-dice cases at the cost of extra compute. "
             "Run the integration test in both modes to compare end-to-end accuracy directly.",
    )
    args = parser.parse_args()

    labels_dir = args.labels_dir or (args.images_dir.parent / "labels")
    output_dir = args.output or (args.checkpoint.parent / "integration_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")

    image_paths = sorted(
        p for p in args.images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {args.images_dir}")
    print(f"Found {len(image_paths)} raw photo(s) in {args.images_dir}")

    stage1_model = load_model(args.checkpoint, device)
    stage1_transform = build_test_transform()
    pipeline = DiceInferencePipeline(args.checkpoint, device=str(device))
    stage1_labels, stage1_predictions = [], []
    stage2_matched_labels, stage2_matched_predictions = [], []

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

        image_stage1_correct = 0
        for gt in gt_boxes:
            crop = crop_ground_truth_box(image_bgr, gt["box"])
            if crop is None:
                continue
            pred_face = stage1_classify(stage1_model, stage1_transform, device, crop)
            stage1_labels.append(gt["label"] - 1)
            stage1_predictions.append(pred_face - 1)
            if pred_face == gt["label"]:
                image_stage1_correct += 1

        detections, _ = pipeline.process_photo(
            image_bgr, expected_dice_count=expected_count, debug=False, pip_mode=args.pip_mode
        )
        pred_boxes = [det.box for det in detections]
        total_pred_boxes += len(pred_boxes)

        matches, matched_gt_idx, matched_pred_idx = match_boxes(
            gt_boxes, pred_boxes, min_iou=args.min_match_iou
        )
        total_matched_gt += len(matched_gt_idx)
        total_matched_pred += len(matched_pred_idx)

        image_stage2_correct = 0
        for gi, pi, iou in matches:
            matched_ious.append(iou)
            det = detections[pi]
            gt_label = gt_boxes[gi]["label"]
            stage2_matched_labels.append(gt_label - 1)
            stage2_matched_predictions.append(det.predicted_face - 1)
            if det.predicted_face == gt_label:
                image_stage2_correct += 1

        missed = expected_count - len(matched_gt_idx)
        hallucinated = len(pred_boxes) - len(matched_pred_idx)

        per_image_details.append({
            "image": img_path.name,
            "expected_dice": expected_count,
            "detected_boxes": len(pred_boxes),
            "matched": len(matched_gt_idx),
            "missed": missed,
            "hallucinated": hallucinated,
            "stage1_correct": image_stage1_correct,
            "stage2_correct_of_matched": image_stage2_correct,
        })

    if skipped_no_labels:
        print(f"Skipped {skipped_no_labels} image(s) with no ground-truth label file")

    stage1_labels_arr = np.array(stage1_labels)
    stage1_predictions_arr = np.array(stage1_predictions)
    stage1_accuracy = (
        float((stage1_labels_arr == stage1_predictions_arr).mean())
        if len(stage1_labels_arr) > 0 else None
    )

    stage2_matched_labels_arr = np.array(stage2_matched_labels)
    stage2_matched_predictions_arr = np.array(stage2_matched_predictions)
    stage2_correct_matched = int((stage2_matched_labels_arr == stage2_matched_predictions_arr).sum())
    end_to_end_accuracy = stage2_correct_matched / total_gt_dice if total_gt_dice > 0 else None

    accuracy_among_matched = (
        stage2_correct_matched / total_matched_gt if total_matched_gt > 0 else None
    )

    detection_recall = total_matched_gt / total_gt_dice if total_gt_dice > 0 else None
    hallucination_rate = (
        (total_pred_boxes - total_matched_pred) / total_pred_boxes if total_pred_boxes > 0 else None
    )

    mean_matched_iou = float(np.mean(matched_ious)) if matched_ious else None

    localization_tax = (
        stage1_accuracy - end_to_end_accuracy
        if stage1_accuracy is not None and end_to_end_accuracy is not None else None
    )

    print(f"\n{'=' * 60}")
    print("  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Ground-truth dice evaluated:     {total_gt_dice}")
    print(f"  Stage 1 accuracy (GT crops):      {stage1_accuracy:.4f}" if stage1_accuracy is not None else "  Stage 1 accuracy: N/A")
    print(f"  Detection recall (locate_dice):   {detection_recall:.4f}" if detection_recall is not None else "  Detection recall: N/A")
    print(f"  Hallucination rate:               {hallucination_rate:.4f}" if hallucination_rate is not None else "  Hallucination rate: N/A")
    print(f"  Mean IoU (matched boxes):          {mean_matched_iou:.4f}" if mean_matched_iou is not None else "  Mean matched IoU: N/A")
    print(f"  Accuracy among matched dice:       {accuracy_among_matched:.4f}" if accuracy_among_matched is not None else "  Accuracy among matched: N/A")
    print(f"  End-to-end accuracy (all GT dice): {end_to_end_accuracy:.4f}" if end_to_end_accuracy is not None else "  End-to-end accuracy: N/A")
    print(f"  LOCALIZATION TAX:                  {localization_tax:.4f}" if localization_tax is not None else "  Localization tax: N/A")
    print(f"{'=' * 60}")

    if len(stage1_labels_arr) > 0:
        cm1 = confusion_matrix(stage1_labels_arr, stage1_predictions_arr, NUM_CLASSES)
        p1, r1, f1_1, s1 = per_class_performance(cm1)
        write_classification_report(
            CLASS_NAMES, p1, r1, f1_1, s1, output_dir / "stage1_classification_report.txt"
        )
        plot_confusion_matrices(cm1, CLASS_NAMES, output_dir / "stage1_confusion_matrix.png")

    if len(stage2_matched_labels_arr) > 0:
        cm2 = confusion_matrix(stage2_matched_labels_arr, stage2_matched_predictions_arr, NUM_CLASSES)
        p2, r2, f2_2, s2 = per_class_performance(cm2)
        write_classification_report(
            CLASS_NAMES, p2, r2, f2_2, s2, output_dir / "stage2_classification_report.txt"
        )
        plot_confusion_matrices(cm2, CLASS_NAMES, output_dir / "stage2_confusion_matrix.png")

    metrics = {
        "checkpoint": str(args.checkpoint),
        "pip_mode": args.pip_mode,
        "images_dir": str(args.images_dir),
        "num_images_evaluated": len(image_paths) - skipped_no_labels,
        "min_match_iou": args.min_match_iou,
        "total_ground_truth_dice": total_gt_dice,
        "stage1_accuracy": stage1_accuracy,
        "detection_recall": detection_recall,
        "hallucination_rate": hallucination_rate,
        "mean_matched_iou": mean_matched_iou,
        "accuracy_among_matched_dice": accuracy_among_matched,
        "end_to_end_accuracy": end_to_end_accuracy,
        "localization_tax": localization_tax,
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