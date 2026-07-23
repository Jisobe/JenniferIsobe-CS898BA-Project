import argparse
import cv2 as cv
import numpy as np
from pathlib import Path

BLUR_KERNEL = (5, 5)
SIGMA = .5
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8,8)
MORPH_KERNEL_SIZE = (5, 5)
MIN_AREA_FRACTION = 0.0015
MAX_AREA_FRACTION = 0.10
MIN_ASPECT_RATIO = 0.45
MAX_ASPECT_RATIO = 2.2
MIN_SOLIDITY = 0.75
CHANNEL = "l"
# CHANNEL = "saturation"
VALID_CHANNELS = ("l", "saturation")
CURRENT_DIR = Path.cwd()
OUTPUT_DIR = CURRENT_DIR / "runs/localization/run_10"
IMAGE_DIR = CURRENT_DIR / "data/raw/d6_ - 8.jpeg"
DEBUG = True

def contour_solidity(contour):
    area = cv.contourArea(contour)
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    if hull_area == 0:
        return 0
    return area / hull_area

def filter_dice_contours(contours, image_area):
    min_area_px = MIN_AREA_FRACTION * image_area
    max_area_px = MAX_AREA_FRACTION * image_area
    boxes = []
    for c in contours:
        area = cv.contourArea(c)
        if area < min_area_px or area > max_area_px:
            continue

        x, y, w, h = cv.boundingRect(c)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            continue

        if contour_solidity(c) < MIN_SOLIDITY:
            continue

        boxes.append((x, y, w, h))

    return boxes

def locate_dice(image_bgr, debug=False):
    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    grey = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    # blurred = cv.GaussianBlur(grey, BLUR_KERNEL, SIGMA)
    blurred = cv.medianBlur(grey,7)
    # clahe = cv.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    # if CHANNEL == "l":
    #     lab = cv.cvtColor(blurred, cv.COLOR_BGR2LAB)
    #     raw_channel, _, _ = cv.split(lab)
    # else:
    #     hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    #     _, raw_channel, _ = cv.split(hsv)
    # normalized_channel = clahe.apply(raw_channel)
    # _, threshold = cv.threshold(normalized_channel, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, threshold = cv.threshold(blurred, 130, 255, cv.THRESH_BINARY)
    edges = cv.Canny(blurred,threshold1=25, threshold2=50, apertureSize=3)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    opened = cv.morphologyEx(threshold, cv.MORPH_OPEN, kernel, iterations=3)
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxes = filter_dice_contours(contours, image_area)
    circled = image_bgr.copy()
    circles = cv.HoughCircles(
        threshold,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=image_bgr.shape[0]/85,
        param1=200,
        param2=10,
        minRadius=20,
        maxRadius=30
    )

    # Draw only the first detected circle
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv.circle(circled, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv.circle(circled, (i[0], i[1]), 2, (0, 0, 255), 3)

    debug_images = None
    if debug:
        debug_images = {
            "blurred": blurred,
            # "normalized_channel": normalized_channel,
            "edges": edges,
            "grey": grey,
            "threshold": threshold,
            "closed": closed,
            "circled": circled
        }

    return boxes, debug_images

def draw_boxes(image_bgr, boxes):
    annotated = image_bgr.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 12)
    return annotated

def main():
    image = cv.imread(IMAGE_DIR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_DIR}")

    boxes, debug_images = locate_dice(image, DEBUG)

    print(f"Detected {len(boxes)} candidate die region(s):")
    for i, box in enumerate(boxes):
        print(f"  die {i}: x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated = draw_boxes(image, boxes)
    cv.imwrite(str(out_dir / "annotated.jpg"), annotated)
    print(f"\nAnnotated image saved to {out_dir / 'annotated.jpg'}")

    if DEBUG and debug_images:
        for name, img in debug_images.items():
            cv.imwrite(str(out_dir / f"debug_{name}.jpg"), img)
        print(f"Debug stage images saved to {out_dir}/debug_*.jpg")

if __name__ == "__main__":
    main()