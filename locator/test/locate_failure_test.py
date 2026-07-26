import argparse
import json
from pathlib import Path
import cv2 as cv
from upgraded_locate_dice import locate_dice, box_iou

MIN_MATCH_IOU = 0.3
GREEN = (0, 200, 0) # ground truth box color
RED = (0, 0, 220) # hallucinated / unmatched prediction box color
YELLOW = (0, 220, 220) # matched prediction box color

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

def match_boxes(gt_boxes, pred_boxes, min_iou=MIN_MATCH_IOU):
    candidates = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            iou = box_iou(gt, pred)
            if iou >= min_iou:
                candidates.append((iou, gi, pi))
    candidates.sort(key=lambda c: -c[0])

    matched_gt, matched_pred = set(), set()
    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
    return matched_gt, matched_pred

def diagnose(expected, detected, matched):
    if matched == expected:
        return "OK"
    if detected == 0:
        return "TOTAL MISS -- box + pip-clustering fallback both found nothing"
    if detected < expected and matched == 0:
        return "UNDERCOUNT + WRONG PLACE -- found too few boxes and none overlap actual dice"
    if detected < expected:
        return "UNDERCOUNT -- found too few boxes but the ones found are at least partly right"
    if detected >= expected and matched == 0:
        return "WRONG PLACE -- found the expected COUNT but none overlap actual dice"
    return "PARTIAL -- some real dice missed"

def draw_diagnostic(image_bgr, gt_boxes, pred_boxes, matched_gt, matched_pred):
    annotated = image_bgr.copy()
    line_thickness = max(4, int(image_bgr.shape[0] / 400))

    for gi, (x, y, w, h) in enumerate(gt_boxes):
        cv.rectangle(annotated, (x, y), (x + w, y + h), GREEN, line_thickness)
        cv.putText(annotated, f"GT{gi}", (x, max(0, y - 8)),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 2, cv.LINE_AA)

    for pi, (x, y, w, h) in enumerate(pred_boxes):
        color = YELLOW if pi in matched_pred else RED
        thickness = line_thickness if pi in matched_pred else max(2, line_thickness - 2)
        cv.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
        label = f"P{pi}" + (" (matched)" if pi in matched_pred else " (hallucinated)")
        cv.putText(annotated, label, (x, y + h + 28),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv.LINE_AA)

    return annotated

def pick_worst_images(sweep_json_path: Path, top_n: int):
    summary = json.loads(sweep_json_path.read_text())
    per_image = summary["per_image"]

    def deficit(entry):
        return (-entry["missed"], -entry["hallucinated"])

    ranked = sorted(per_image, key=deficit)
    return [entry["image"] for entry in ranked[:top_n]]

def main():
    parser = argparse.ArgumentParser(description="Visually triage locate_dice() failures")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, default=None,
                         help="Default: <images-dir>/../labels")
    parser.add_argument("--sweep-json", type=Path, default=None,
                         help="Path to a prior localization_sweep.json to auto-pick worst images from")
    parser.add_argument("--top-n", type=int, default=10,
                         help="How many worst images to pull from --sweep-json (default: 10)")
    parser.add_argument("--image-names", nargs="+", default=None,
                         help="Specific image filenames to diagnose instead of using --sweep-json")
    parser.add_argument("--output", type=Path, default=Path("runs/localization/failure_triage"))
    parser.add_argument("--min-iou", type=float, default=MIN_MATCH_IOU)
    args = parser.parse_args()

    if not args.sweep_json and not args.image_names:
        parser.error("Provide either --sweep-json (to auto-pick worst images) or --image-names")

    labels_dir = args.labels_dir or (args.images_dir.parent / "labels")
    args.output.mkdir(parents=True, exist_ok=True)

    if args.image_names:
        target_names = args.image_names
    else:
        target_names = pick_worst_images(args.sweep_json, args.top_n)

    print(f"Diagnosing {len(target_names)} image(s), output -> {args.output}\n")

    for name in target_names:
        img_path = args.images_dir / name
        image_bgr = cv.imread(str(img_path))
        if image_bgr is None:
            print(f"  {name}: COULD NOT READ IMAGE at {img_path}")
            continue
        img_h, img_w = image_bgr.shape[:2]

        gt_boxes = load_ground_truth_boxes(labels_dir, img_path.stem, img_w, img_h)
        expected_count = len(gt_boxes)
        pred_boxes, _ = locate_dice(image_bgr, expected_dice_count=expected_count, debug=False)
        matched_gt, matched_pred = match_boxes(gt_boxes, pred_boxes, args.min_iou)

        diagnosis = diagnose(expected_count, len(pred_boxes), len(matched_gt))
        print(
            f"  {name}\n"
            f"    expected={expected_count} detected={len(pred_boxes)} matched={len(matched_gt)}\n"
            f"    diagnosis: {diagnosis}"
        )

        annotated = draw_diagnostic(image_bgr, gt_boxes, pred_boxes, matched_gt, matched_pred)
        out_path = args.output / f"{img_path.stem}_diagnostic.jpg"
        cv.imwrite(str(out_path), annotated)
        print(f"    saved: {out_path}\n")

    print(f"Done. Open the images in {args.output} -- "
          f"green = ground truth, yellow = matched prediction, red = hallucinated prediction.")


if __name__ == "__main__":
    main()