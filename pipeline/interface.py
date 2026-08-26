from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from ultralytics import YOLO


@dataclass
class DieDetection:
    box: tuple[int, int, int, int]  # x, y, w, h in original photo coords (top-left corner)
    predicted_face: int             # 1-6
    confidence: float               # YOLO detection confidence for this box


class DiceInferencePipeline:
    """Single-stage pipeline: one YOLO forward pass does both localization
    and classification, replacing the old locate_dice() ensemble +
    DiceClassifier crop-and-classify two-stage flow entirely."""

    def __init__(self, weights_path: str | Path):
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {weights_path}. "
                "Point DiceInferencePipeline at your fine-tuned weights "
                "(e.g. runs/yolo/<run_name>/weights/best.pt)."
            )
        self.model = YOLO(str(weights_path))

    def process_photo(self, photo_bgr: np.ndarray,
                       expected_dice_count: int | None = None) -> list[DieDetection]:
        """Detect and classify every die in one pass.

        expected_dice_count: how many dice are actually in this photo (1-5).
        RollCall always knows this deterministically -- 5 on the first roll
        of a turn, or however many weren't kept on a reroll -- so when YOLO
        returns more boxes than expected (a false-positive detection), we
        trust that count and keep only the highest-confidence boxes. If YOLO
        returns fewer than expected, we don't fabricate detections; the
        caller (app.py) surfaces the mismatch so the player can fix it via
        manual override.
        """
        # end2end=False switches from YOLO26's default NMS-free one-to-one
        # head to the traditional one-to-many head + NMS post-processing.
        # The default head is trained to output one clean box per object on
        # its own, but on this dataset it sometimes still emits duplicate
        # boxes on the same die (small-object case, likely undertrained) --
        # iou= has no effect at all on the default path since there's no
        # NMS step to apply it to. This trades a small amount of latency
        # and Ultralytics' own ~0.6-0.8 AP for a real duplicate-suppression
        # knob, which matters more here than raw speed.
        results = self.model.predict(photo_bgr, verbose=False, end2end=False, iou=0.5)[0]
        names = results.names  # class index -> label string, e.g. {0: "1", ..., 5: "6"}

        detections: list[DieDetection] = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_idx = int(box.cls.item())
            conf = float(box.conf.item())
            face = int(names[cls_idx])
            detections.append(DieDetection(
                box=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                predicted_face=face,
                confidence=conf,
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)

        if expected_dice_count is not None and len(detections) > expected_dice_count:
            detections = detections[:expected_dice_count]

        return detections


def draw_annotated(photo_bgr: np.ndarray, detections: list[DieDetection]) -> np.ndarray:
    """Draw boxes + predicted face + confidence on the original photo,
    for the 'here's what I saw' panel in the UI."""
    annotated = photo_bgr.copy()
    for det in detections:
        x, y, w, h = det.box
        cv.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 6)
        label = f"{det.predicted_face} ({det.confidence:.0%})"
        cv.putText(
            annotated, label, (x, max(0, y - 12)),
            cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3, cv.LINE_AA,
        )
    return annotated