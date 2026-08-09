"""
RollCall - Raw Prediction Debugger

Runs YOLO with NMS almost fully disabled (iou=0.99, low conf threshold) on
a single image so you can see every candidate box the model proposed
BEFORE suppression, and measure the actual IoU between any two boxes that
look like duplicates. Use this when test_yolo.py results don't change
after adjusting the `iou` kwarg in interface.py -- it tells you whether
the duplicate boxes are close enough for NMS to ever merge them, or
whether this is a box-regression-precision problem instead.

Usage:
    uv run debug_raw_predictions.py --weights runs/yolo/run_01/weights/best.pt --image "data/labeled/test/images/d6_ - 104_jpeg.rf.DevGuHIB2YBZphSPSIU2.jpeg"
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def box_iou_xyxy(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.15, help="Low confidence floor so weak duplicate candidates aren't dropped before we can see them")
    parser.add_argument("--target-iou", type=float, default=0.5, help="The NMS iou threshold you're actually using in interface.py, to verify it really suppresses the duplicate")
    args = parser.parse_args()

    model = YOLO(str(args.weights))

    def run(iou_value):
        results = model.predict(str(args.image), conf=args.conf, iou=iou_value, verbose=False)[0]
        names = results.names
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_idx = int(box.cls.item())
            conf = float(box.conf.item())
            boxes.append({"xyxy": (x1, y1, x2, y2), "cls_idx": cls_idx, "face": names[cls_idx], "conf": conf})
        boxes.sort(key=lambda b: -b["conf"])
        return boxes

    print(f"\n--- Raw candidates (iou=0.99, NMS effectively disabled) ---")
    raw_boxes = run(0.99)
    print(f"{len(raw_boxes)} box(es):\n")
    for i, b in enumerate(raw_boxes):
        x1, y1, x2, y2 = b["xyxy"]
        print(f"  [{i}] cls_idx={b['cls_idx']}  face={b['face']}  conf={b['conf']:.2%}  box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")

    print(f"\nPairwise IoU for same-class box pairs:")
    found_any = False
    for i in range(len(raw_boxes)):
        for j in range(i + 1, len(raw_boxes)):
            if raw_boxes[i]["cls_idx"] != raw_boxes[j]["cls_idx"]:
                continue
            iou = box_iou_xyxy(raw_boxes[i]["xyxy"], raw_boxes[j]["xyxy"])
            if iou > 0.05:
                found_any = True
                print(f"  [{i}] vs [{j}]  (cls_idx={raw_boxes[i]['cls_idx']})  IoU = {iou:.3f}")
    if not found_any:
        print("  No overlapping same-class pairs found above 0.05 IoU.")

    print(f"\n--- After NMS at iou={args.target_iou} (your interface.py setting) ---")
    filtered_boxes = run(args.target_iou)
    print(f"{len(filtered_boxes)} box(es) survive:\n")
    for i, b in enumerate(filtered_boxes):
        x1, y1, x2, y2 = b["xyxy"]
        print(f"  [{i}] cls_idx={b['cls_idx']}  face={b['face']}  conf={b['conf']:.2%}  box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")

    if len(filtered_boxes) == len(raw_boxes):
        print(f"\n  NOTHING was suppressed at iou={args.target_iou} despite high-IoU same-class pairs above.")
        print(f"  This means NMS itself isn't the lever to pull here -- something else is going on")
        print(f"  (e.g. an Ultralytics version quirk, or these boxes' IoU is computed differently")
        print(f"  than the naive xyxy IoU this script uses -- worth checking your installed")
        print(f"  ultralytics version: `uv run python -c \"import ultralytics; print(ultralytics.__version__)\"`)")
    else:
        print(f"\n  {len(raw_boxes) - len(filtered_boxes)} box(es) suppressed -- NMS at iou={args.target_iou} DOES work on this image.")
        print(f"  If test_yolo.py still isn't reflecting this, the problem is upstream of NMS")
        print(f"  (stale import, uncommitted edit, or a different weights/interface.py being loaded).")


if __name__ == "__main__":
    main()