"""
RollCall - Detection Error Visualizer

Draws ground-truth boxes and YOLO's predicted boxes on top of each other so
you can actually see what's happening in the images test_yolo.py flags as
having a missed die and/or a hallucinated box. Numbers alone can't tell you
whether a "miss + hallucination" pair in the same image is:

  - YOLO putting two boxes on the same die (duplicate detection) while
    missing a different die elsewhere in the frame, or
  - A genuinely undetected die (edge of frame, occlusion, glare) plus an
    unrelated false-positive box somewhere else in that same photo.

Color code drawn on each output image:
  GREEN  solid  - ground-truth box that WAS matched to a prediction
  BLUE   solid  - predicted box that matched a ground-truth box (drawn
                   slightly inset from the green box so both are visible
                   when they overlap almost perfectly)
  RED    solid  - ground-truth box with NO matching prediction (missed die)
  ORANGE solid  - predicted box with NO matching ground truth (hallucination)

By default, only visualizes images that actually have a miss and/or
hallucination (same logic test_yolo.py uses) -- no need to manually copy
filenames out of per_image_details.json first.

Usage:
    uv run overlay_errors.py --weights runs/yolo/run_01/weights/best.pt
    uv run overlay_errors.py --weights runs/yolo/run_01/weights/best.pt --images "d6_ - 18_jpeg"
"""

import argparse
from pathlib import Path

import cv2 as cv

from interface import DiceInferencePipeline
from test_yolo import load_ground_truth, match_boxes, MIN_MATCH_IOU

GREEN = (0, 200, 0)
BLUE = (255, 120, 0)
RED = (0, 0, 255)
ORANGE = (0, 140, 255)


def draw_box(img, box, color, label, thickness=4, inset=0):
    x, y, w, h = box
    x, y, w, h = x + inset, y + inset, w - 2 * inset, h - 2 * inset
    cv.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    cv.putText(
        img, label, (x, max(0, y - 10)),
        cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv.LINE_AA,
    )


def draw_legend(img):
    entries = [
        ("GT matched", GREEN), ("Pred matched", BLUE),
        ("GT MISSED", RED), ("Pred HALLUCINATED", ORANGE),
    ]
    y = 40
    for label, color in entries:
        cv.rectangle(img, (20, y - 20), (50, y), color, -1)
        cv.putText(img, label, (60, y - 3), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv.LINE_AA)
        y += 40


def visualize_image(img_path, labels_dir, pipeline, output_dir, min_iou):
    image_bgr = cv.imread(str(img_path))
    if image_bgr is None:
        print(f"  Could not read image, skipping: {img_path.name}")
        return None

    img_h, img_w = image_bgr.shape[:2]
    gt_boxes = load_ground_truth(labels_dir, img_path.stem, img_w, img_h)
    if not gt_boxes:
        return None

    expected_count = len(gt_boxes)
    detections = pipeline.process_photo(image_bgr, expected_dice_count=expected_count)
    pred_boxes = [det.box for det in detections]

    matches, matched_gt_idx, matched_pred_idx = match_boxes(gt_boxes, pred_boxes, min_iou=min_iou)
    missed = expected_count - len(matched_gt_idx)
    hallucinated = len(pred_boxes) - len(matched_pred_idx)

    if missed == 0 and hallucinated == 0:
        return None  # nothing wrong with this image, skip it

    annotated = image_bgr.copy()

    for gi, pi, iou in matches:
        gt = gt_boxes[gi]
        det = detections[pi]
        draw_box(annotated, gt["box"], GREEN, f"GT:{gt['label']}")
        draw_box(annotated, det.box, BLUE, f"pred:{det.predicted_face} ({det.confidence:.0%})", inset=6)

    for gi, gt in enumerate(gt_boxes):
        if gi not in matched_gt_idx:
            draw_box(annotated, gt["box"], RED, f"MISSED (true={gt['label']})", thickness=5)

    for pi, det in enumerate(detections):
        if pi not in matched_pred_idx:
            draw_box(annotated, det.box, ORANGE, f"HALLUC pred:{det.predicted_face} ({det.confidence:.0%})", thickness=5)

    draw_legend(annotated)

    out_path = output_dir / f"{img_path.stem}_annotated.jpg"
    cv.imwrite(str(out_path), annotated)
    print(f"  {img_path.name}: missed={missed}, hallucinated={hallucinated} -> {out_path.name}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Visualize ground-truth vs. predicted boxes for error cases")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=Path("data/labeled/test/images"))
    parser.add_argument("--labels-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Default: <weights run dir>/error_visualizations",
    )
    parser.add_argument(
        "--images", type=str, default=None,
        help="Comma-separated substrings to filter to specific images (e.g. '18,236'). "
             "Default: visualize every image with a miss or hallucination.",
    )
    parser.add_argument("--min-match-iou", type=float, default=MIN_MATCH_IOU)
    args = parser.parse_args()

    labels_dir = args.labels_dir or (args.images_dir.parent / "labels")
    output_dir = args.output_dir or (args.weights.parent.parent / "error_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in args.images_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if args.images:
        filters = [f.strip() for f in args.images.split(",")]
        image_paths = [p for p in image_paths if any(f in p.name for f in filters)]

    if not image_paths:
        raise RuntimeError(f"No matching images found under {args.images_dir}")

    pipeline = DiceInferencePipeline(args.weights)

    print(f"Scanning {len(image_paths)} image(s) for detection errors...\n")
    flagged = 0
    for img_path in image_paths:
        result = visualize_image(img_path, labels_dir, pipeline, output_dir, args.min_match_iou)
        if result is not None:
            flagged += 1

    print(f"\n{flagged} image(s) with errors visualized. Saved to: {output_dir}")


if __name__ == "__main__":
    main()