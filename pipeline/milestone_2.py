# This file was used for the milestone 2 presentation demo and may not be updated with newer functionality
import argparse
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from locator.locate_dice import locate_dice
from locator.crop_script import apply_clahe, add_padding
from classifier.cnn_dice_detection_model import DiceClassifier

CLASS_NAMES = ["1", "2", "3", "4", "5", "6"]
IMG_SIZE = 64
CURRENT_DIR = Path.cwd()
OUTPUT_DIR = CURRENT_DIR / "runs/pipeline/run_1"
IMAGE_DIR = CURRENT_DIR / "data/raw/d6_ - 1.jpeg"
MODEL_PATH = CURRENT_DIR / "runs/custom/run_19/best_model.pt"
NORM_MEAN = [0.5096, 0.4681, 0.3996]
NORM_STD = [0.2762, 0.2643, 0.2510]

def load_model(model_path, device):
    model = DiceClassifier(classes=6)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def crop_and_normalize(image_bgr, box, padding, apply_normalization):
    height, width = image_bgr.shape[:2]
    x, y, w, h = box
    x1, y1, x2, y2 = x, y, x + w, y + h
    x1, y1, x2, y2 = add_padding(x1, y1, x2, y2, padding, width, height)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    if apply_normalization:
        crop = apply_clahe(crop)

    crop_resized = cv.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv.INTER_LINEAR)
    return crop_resized

def crop_to_tensor(crop_bgr, device):
    crop_rgb = cv.cvtColor(crop_bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1)
    mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(NORM_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device)

def classify_crop(model, crop_bgr, device):
    tensor = crop_to_tensor(crop_bgr, device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        confidence, predicted_idx = probs.max(dim=1)
    predicted_label = CLASS_NAMES[predicted_idx.item()]
    return predicted_label, confidence.item()

def draw_annotated_photo(image_bgr, results):
    annotated = image_bgr.copy()
    for (x, y, w, h), label, confidence in results:
        cv.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 12)
        text = f"{label} ({confidence * 100:.0f}%)"
        text_y = max(y - 15, 30)
        cv.putText(annotated, text, (x, text_y), cv.FONT_HERSHEY_SIMPLEX,
                   5, (0, 255, 0), 3, cv.LINE_AA)
    return annotated

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    image_path = Path(IMAGE_DIR)
    model_path = Path(MODEL_PATH)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        sys.exit(f"Could not find image: {image_path}")
    if not model_path.exists():
        sys.exit(f"Could not find model checkpoint: {model_path}")

    image = cv.imread(str(image_path))
    if image is None:
        sys.exit(f"OpenCV could not read image: {image_path}")

    print("\n[Stage 1] Locating dice with locate_dice.py...")
    boxes, _ = locate_dice(image, debug=False)
    print(f"  Detected {len(boxes)} candidate die region(s)")
    if len(boxes) == 0:
        sys.exit("No dice localized in this image - try a different photo for the demo.")

    print("\n[Stage 2] Loading DiceClassifier checkpoint...")
    model = load_model(model_path, device)

    print("\n[Stage 3] Cropping + classifying each detected die...")
    apply_normalization = True
    results = []
    crops = []
    for i, box in enumerate(boxes):
        crop = crop_and_normalize(image, box, 8, apply_normalization)
        if crop is None:
            print(f"  die {i}: empty crop, skipping")
            continue
        label, confidence = classify_crop(model, crop, device)
        results.append((box, label, confidence))
        crops.append(crop)
        x, y, w, h = box
        print(f"  die {i}: box=({x},{y},{w},{h}) -> predicted {label} "
              f"({confidence * 100:.1f}% confidence)")

    if not results:
        sys.exit("All detected boxes produced empty crops - nothing to classify.")

    print("\n[Output] Saving annotated photo...")
    annotated = draw_annotated_photo(image, results)
    annotated_path = output_dir / "annotated_photo.jpg"
    cv.imwrite(str(annotated_path), annotated)
    print(f"  Annotated photo: {annotated_path}")

    print("\n[Summary]")
    values = [label for _, label, _ in results]
    print(f"  Detected roll: {', '.join(values)}")

if __name__ == "__main__":
    main()