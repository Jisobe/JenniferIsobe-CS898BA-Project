from dataclasses import dataclass
from pathlib import Path
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from classifier.cnn_dice_detection_model import DiceClassifier
from locator.upgraded_locate_dice import locate_dice

NORM_MEAN = [0.5096, 0.4681, 0.3996]
NORM_STD  = [0.2762, 0.2643, 0.2510]
INPUT_SIZE = 64
CLASS_NAMES = ["1", "2", "3", "4", "5", "6"]

@dataclass
class DieDetection:
    box: tuple[int, int, int, int]
    predicted_face: int
    confidence: float
    class_probs: list[float]

def _apply_clahe(crop_bgr):
    lab = cv.cvtColor(crop_bgr, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv.merge((l, a, b))
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)

def preprocess_to_tensor(rgb_uint8):
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) uint8 image, got shape {rgb_uint8.shape}")

    resized = cv.resize(rgb_uint8, (INPUT_SIZE, INPUT_SIZE), interpolation=cv.INTER_AREA)
    chw = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    mean = np.array(NORM_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(NORM_STD, dtype=np.float32).reshape(3, 1, 1)
    chw = (chw - mean) / std

    return torch.from_numpy(chw)


class DiceInferencePipeline:
    def __init__(self, checkpoint_path, device= None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DiceClassifier(classes=len(CLASS_NAMES))
        self._load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model not found at {checkpoint_path}. "
                "Point DiceInferencePipeline at your trained weights (e.g. runs/custom/<run_name>/best_model.pt)."
            )
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)

    def classify_crop(self, crop_bgr):
        prepped = _apply_clahe(crop_bgr)
        prepped = cv.cvtColor(prepped, cv.COLOR_BGR2RGB)
        tensor = preprocess_to_tensor(prepped).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

        pred_idx = int(np.argmax(probs))
        return int(CLASS_NAMES[pred_idx]), float(probs[pred_idx]), probs

    def process_photo(self, photo_bgr, expected_dice_count = None, debug= False, pip_mode = "always"):
        slots, debug_images, candidate_groups = locate_dice(
            photo_bgr, expected_dice_count=expected_dice_count, debug=debug, pip_mode=pip_mode,
            return_candidates=True
        )
        detections= []
        for candidate_boxes in candidate_groups:
            detection = self._select_best_candidate(photo_bgr, candidate_boxes)
            if detection is not None:
                detections.append(detection)

        return detections, debug_images

    def _select_best_candidate(self, photo_bgr, candidate_boxes):
        best = None
        for (x, y, w, h) in candidate_boxes:
            crop = photo_bgr[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            face, conf, probs = self.classify_crop(crop)
            if best is None or conf > best.confidence:
                best = DieDetection(
                    box=(x, y, w, h),
                    predicted_face=face,
                    confidence=conf,
                    class_probs=probs,
                )
        return best

def draw_annotated(photo_bgr, detections):
    annotated = photo_bgr.copy()
    for det in detections:
        x, y, w, h = det.box
        cv.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 6)
        label = f"{det.predicted_face} ({det.confidence:.0%})"
        cv.putText(
            annotated, label, (x, max(0, y - 12)),
            cv.FONT_HERSHEY_SIMPLEX, 6, (0, 200, 0), 3, cv.LINE_AA,
        )
    return annotated