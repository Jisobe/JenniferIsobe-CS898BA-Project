import argparse
import functools
import json
import time
from pathlib import Path
import cv2 as cv
import numpy as np
from locator.upgraded_locate_dice import locate_dice, box_iou

print = functools.partial(print, flush=True)

MIN_MATCH_IOU = 0.3

def load_ground_truth_boxes(labels_dir, img_stem, img_w, img_h):
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
            _, cx, cy, bw, bh = (float(p) if i > 0 else int(p) for i, p in enumerate(parts))
        except ValueError:
            continue

        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes

def get_labels_dir(image_path):
    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        labels_parts = parts[:idx] + ["labels"] + parts[idx + 1:-1]
        return Path(*labels_parts)
    return image_path.parent / "labels"

def match_boxes(gt_boxes, pred_boxes, min_iou=MIN_MATCH_IOU):
    candidates = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            iou = box_iou(gt, pred)
            if iou >= min_iou:
                candidates.append((iou, gi, pi))
    candidates.sort(key=lambda c: -c[0])

    matched_gt, matched_pred = set(), set()
    ious = []
    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        ious.append(iou)

    return matched_gt, matched_pred, ious

def evaluate_localization(images_dir: Path, labels_dir: Path, min_iou: float, limit: int = None, pip_mode: str = "fallback"):
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    if limit is not None:
        image_paths = image_paths[:limit]

    total_images = len(image_paths)
    print(
        f" Processing {total_images} image(s). Each full-resolution photo runs a 4-way color space/threshold plus a potential pip-clustering fallback pass, "
        f"so expect about a few seconds to ~20s per image depending on photo resolution and device -- progress below, one line per image."
    )

    total_gt = 0
    total_matched_gt = 0
    total_pred = 0
    total_matched_pred = 0
    all_ious = []
    count_mismatches = 0
    per_image = []
    skipped = 0
    start_time = time.monotonic()

    for i, img_path in enumerate(image_paths, start=1):
        image_start = time.monotonic()
        image_bgr = cv.imread(str(img_path))
        if image_bgr is None:
            print(f"  [{i}/{total_images}] Could not read image, skipping: {img_path.name}")
            continue
        img_h, img_w = image_bgr.shape[:2]

        gt_boxes = load_ground_truth_boxes(labels_dir, img_path.stem, img_w, img_h)
        if not gt_boxes:
            skipped += 1
            print(f"  [{i}/{total_images}] {img_path.name}: no label file, skipped")
            continue

        expected_count = len(gt_boxes)
        pred_boxes, _ = locate_dice(image_bgr, expected_dice_count=expected_count, debug=False, pip_mode=pip_mode)
        matched_gt, matched_pred, ious = match_boxes(gt_boxes, pred_boxes, min_iou)
        total_gt += len(gt_boxes)
        total_matched_gt += len(matched_gt)
        total_pred += len(pred_boxes)
        total_matched_pred += len(matched_pred)
        all_ious.extend(ious)

        count_mismatch = len(pred_boxes) != expected_count
        if count_mismatch:
            count_mismatches += 1

        per_image.append({
            "image": img_path.name,
            "expected_dice": expected_count,
            "detected_boxes": len(pred_boxes),
            "matched": len(matched_gt),
            "missed": expected_count - len(matched_gt),
            "hallucinated": len(pred_boxes) - len(matched_pred),
            "count_mismatch": count_mismatch,
        })

        image_elapsed = time.monotonic() - image_start
        elapsed_total = time.monotonic() - start_time
        avg_per_image = elapsed_total / i
        remaining = avg_per_image * (total_images - i)
        print(
            f"  [{i}/{total_images}] {img_path.name}: "
            f"expected={expected_count} detected={len(pred_boxes)} matched={len(matched_gt)} "
            f"({image_elapsed:.1f}s, ~{remaining:.0f}s remaining)"
        )

    if skipped:
        print(f"Skipped {skipped} image(s) with no ground-truth label file")

    n_images = len(per_image)
    return {
        "num_images": n_images,
        "pip_mode": pip_mode,
        "total_ground_truth_dice": total_gt,
        "detection_recall": total_matched_gt / total_gt if total_gt else None,
        "hallucination_rate": (total_pred - total_matched_pred) / total_pred if total_pred else None,
        "mean_matched_iou": float(np.mean(all_ious)) if all_ious else None,
        "count_mismatch_rate": count_mismatches / n_images if n_images else None,
        "per_image": per_image,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Standalone localization test for locate_dice() -- no classifier required"
    )
    parser.add_argument(
        "--images-dir", type=Path, default=None,
        help="Raw full-photo images with YOLO labels for a full localization sweep",
    )
    parser.add_argument(
        "--labels-dir", type=Path, default=None,
        help="YOLO label directory (default: <images-dir>/../labels)",
    )
    parser.add_argument(
        "--min-iou", type=float, default=MIN_MATCH_IOU,
        help=f"Minimum IoU to count a detection as matching ground truth (default: {MIN_MATCH_IOU})",
    )
    parser.add_argument(
        "--pip-mode", choices=["fallback", "always"], default="fallback",
        help="'fallback' (default): pip-clustering only runs to top up a shortfall, matching "
             "the original tested behavior. 'always': pip-clustering runs unconditionally and "
             "its candidates vote alongside the silhouette ensemble -- targets touching-dice "
             "cases at the cost of extra compute per image. Run the same sweep/regression "
             "suite in both modes to compare recall/hallucination/IoU numbers directly.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N images in --images-dir -- useful for a quick "
             "sanity check / timing estimate before committing to a full sweep",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/localization/test_results"),
        help="Where to save the full-sweep results JSON",
    )
    args = parser.parse_args()

    if not args.images_dir and not args.regression_suite:
        parser.error("Provide --images-dir for a full sweep and/or --regression-suite for named cases")

    if args.images_dir:
        labels_dir = args.labels_dir or (args.images_dir.parent / "labels")
        print(f"\n{'=' * 60}")
        print(f"  FULL LOCALIZATION SWEEP: {args.images_dir}")
        print(f"{'=' * 60}")
        sweep_start = time.monotonic()
        summary = evaluate_localization(
            args.images_dir, labels_dir, args.min_iou, limit=args.limit, pip_mode=args.pip_mode
        )
        sweep_elapsed = time.monotonic() - sweep_start

        print(f"  Images evaluated:      {summary['num_images']}")
        print(f"  Total time:            {sweep_elapsed:.1f}s ({sweep_elapsed / max(summary['num_images'], 1):.1f}s/image avg)")
        print(f"  Ground-truth dice:     {summary['total_ground_truth_dice']}")
        recall = summary["detection_recall"]
        halluc = summary["hallucination_rate"]
        iou = summary["mean_matched_iou"]
        mismatch = summary["count_mismatch_rate"]
        print(f"  Detection recall:      {recall:.4f}" if recall is not None else "  Detection recall: N/A")
        print(f"  Hallucination rate:    {halluc:.4f}" if halluc is not None else "  Hallucination rate: N/A")
        print(f"  Mean matched IoU:      {iou:.4f}" if iou is not None else "  Mean matched IoU: N/A")
        print(f"  Count-mismatch rate:   {mismatch:.4f}" if mismatch is not None else "  Count-mismatch rate: N/A")
        print(f"{'=' * 60}")

        args.output.mkdir(parents=True, exist_ok=True)
        out_path = args.output / "localization_sweep.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()