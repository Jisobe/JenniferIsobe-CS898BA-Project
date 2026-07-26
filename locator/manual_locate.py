import argparse
from dataclasses import dataclass
from pathlib import Path
import cv2 as cv
import numpy as np

# image; lighting; dice and background; best dice detection
# 1; good lighting; light dice light bg; grey_hsv_circled, grey_hsv_canny_circled
# 9; good lighting; dark dice light bg; hsv_otsu_boxed, hsv_tresh_boxed, hsv_triangle_boxed
# 22; good lighting; dark dice dark bg; hsv_otsu_boxed, hsv_tresh_boxed, hsv_triangle_boxed, lab_trsh_box, lab_triangle_boxed
# 67; good lighting; light dice dark bg; hsv_otsu_boxed, hsv_triangle_boxed, lab_outsu_boxed, lab_triangle_boxed, lab_tresh_boxed
# 649; poor lighting; dark dice light bg; hsv_otsu_boxed, lab_canny_circled, lab_triangle_boxed
# 661; poor lighting; light dice light bg; grey_hsv_circ
# 728; poor lighting; light dice dark bg; lab_otsu_box (1 box), lab_thresh_boxed (2 boxes), lab_triangle_boxed (2 boes)
# 737; poor lighting; dark dice dark bg; lab_otsu_boxed, lab_triangle_boxed


CURRENT_DIR = Path.cwd()
OUTPUT_DIR = CURRENT_DIR / "runs/localization/manual/run_3"
HSV_DIR = OUTPUT_DIR / "hsv"
LAB_DIR = OUTPUT_DIR / "lab"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
EDGES_DIR = OUTPUT_DIR / "edges"
IMAGE_DIR = CURRENT_DIR / "data/raw/d6_ - 3.jpeg"
DEBUG = True
MIN_AREA_FRACTION = 0.0015
MAX_AREA_FRACTION = 0.10
MIN_ASPECT_RATIO = 0.45
MAX_ASPECT_RATIO = 2.2
MIN_SOLIDITY = 0.75
MORPH_KERNEL_SIZE = (5, 5)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HSV_DIR.mkdir(parents=True, exist_ok=True)
LAB_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
EDGES_DIR.mkdir(parents=True, exist_ok=True)

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

def box_dice(orig_img, input_img):
    image_area = orig_img.shape[0] * orig_img.shape[1]
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    opened = cv.morphologyEx(input_img, cv.MORPH_OPEN, kernel, iterations=3)
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel, iterations=3)
    if len(closed.shape) == 3:
        closed = cv.cvtColor(closed, cv.COLOR_BGR2GRAY)
    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxes = filter_dice_contours(contours, image_area)
    boxed = orig_img.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 12)

    return boxed

def circle_pips(orig_img, input_img):
    circled = orig_img.copy()
    if len(input_img.shape) == 3:
        input_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(
        input_img,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=orig_img.shape[0]/85,
        param1=200,
        param2=10,
        minRadius=20,
        maxRadius=30
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv.circle(circled, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv.circle(circled, (i[0], i[1]), 2, (0, 0, 255), 3)

    return circled

image = cv.imread(IMAGE_DIR)
if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_DIR}")

blurred = cv.GaussianBlur(image, (3,3), .05)

# HSV
hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
grey_hsv = cv.cvtColor(hsv, cv.COLOR_BGR2GRAY)

retval, hsv_thresh_img = cv.threshold(grey_hsv, 127, 255, cv.THRESH_BINARY)
otsu_retval, hsv_otsu_img = cv.threshold(grey_hsv, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
triangle_retval, hsv_triangle_img = cv.threshold(grey_hsv, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)

hsv_adaptive_gaus_img = cv.adaptiveThreshold(
    grey_hsv,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY,
    5,
    2
)

hsv_adaptive_mean_img = cv.adaptiveThreshold(
    grey_hsv,
    255,
    cv.ADAPTIVE_THRESH_MEAN_C,
    cv.THRESH_BINARY,
    5,
    2
)

cv.imwrite(str(HSV_DIR / "original.jpg"), image)
cv.imwrite(str(HSV_DIR / "hsv.jpg"), hsv)
cv.imwrite(str(HSV_DIR / "hsv_grey.jpg"), grey_hsv)
cv.imwrite(str(HSV_DIR / "hsv_thresh.jpg"), hsv_thresh_img)
cv.imwrite(str(HSV_DIR / "hsv_otsu_img.jpg"), hsv_otsu_img)
cv.imwrite(str(HSV_DIR / "hsv_triangle_img.jpg"), hsv_triangle_img)
cv.imwrite(str(HSV_DIR / "hsv_adaptive_gaus_img.jpg"), hsv_adaptive_gaus_img)
cv.imwrite(str(HSV_DIR / "hsv_adaptive_mean_img.jpg"), hsv_adaptive_mean_img)

# Lab
lab = cv.cvtColor(blurred, cv.COLOR_BGR2LAB)
grey_lab = cv.cvtColor(lab, cv.COLOR_BGR2GRAY)

retval, lab_thresh_img = cv.threshold(grey_lab, 127, 255, cv.THRESH_BINARY)
otsu_retval, lab_otsu_img = cv.threshold(grey_lab, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
triangle_retval, lab_triangle_img = cv.threshold(grey_lab, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)

lab_adaptive_gaus_img = cv.adaptiveThreshold(
    grey_lab,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY,
    5,
    2
)

lab_adaptive_mean_img = cv.adaptiveThreshold(
    grey_lab,
    255,
    cv.ADAPTIVE_THRESH_MEAN_C,
    cv.THRESH_BINARY,
    5,
    2
)

cv.imwrite(str(LAB_DIR / "lab.jpg"), lab)
cv.imwrite(str(LAB_DIR / "lab_grey.jpg"), grey_lab)
cv.imwrite(str(LAB_DIR / "lab_thresh_img.jpg"), lab_thresh_img)
cv.imwrite(str(LAB_DIR / "lab_otsu_img.jpg"), lab_otsu_img)
cv.imwrite(str(LAB_DIR / "lab_triangle_img.jpg"), lab_triangle_img)
cv.imwrite(str(LAB_DIR / "lab_adaptive_gaus_img.jpg"), lab_adaptive_gaus_img)
cv.imwrite(str(LAB_DIR / "lab_adaptive_mean_img.jpg"), lab_adaptive_mean_img)

edge_input_images = {
    "hsv": hsv,
    "grey_hsv": grey_hsv,
    "hsv_thresh_img": hsv_thresh_img,
    "hsv_otsu_img": hsv_otsu_img,
    "hsv_triangle_img": hsv_triangle_img,
    "hsv_adaptive_gaus_img": hsv_adaptive_gaus_img,
    "hsv_adaptive_mean_img": hsv_adaptive_mean_img,
    "lab": lab,
    "grey_lab": grey_lab,
    "lab_thresh_img": lab_thresh_img,
    "lab_otsu_img": lab_otsu_img,
    "lab_triangle_img": lab_triangle_img,
    "lab_adaptive_gaus_img": lab_adaptive_gaus_img,
    "lab_adaptive_mean_img": lab_adaptive_mean_img
}

input_images = {
    "hsv": hsv,
    "grey_hsv": grey_hsv,
    "hsv_thresh_img": hsv_thresh_img,
    "hsv_otsu_img": hsv_otsu_img,
    "hsv_triangle_img": hsv_triangle_img,
    "hsv_adaptive_gaus_img": hsv_adaptive_gaus_img,
    "hsv_adaptive_mean_img": hsv_adaptive_mean_img,
    "lab": lab,
    "grey_lab": grey_lab,
    "lab_thresh_img": lab_thresh_img,
    "lab_otsu_img": lab_otsu_img,
    "lab_triangle_img": lab_triangle_img,
    "lab_adaptive_gaus_img": lab_adaptive_gaus_img,
    "lab_adaptive_mean_img": lab_adaptive_mean_img
}

for name, input_image in edge_input_images.items():
    sobel = cv.Sobel(src=input_image, ddepth=cv.CV_8U, dx=1, dy=1, ksize=5)
    canny = cv.Canny(image=input_image, threshold1=75, threshold2=200)
    laplacian = cv.Laplacian(input_image, ddepth=cv.CV_8U, ksize=3)

    cv.imwrite(str(EDGES_DIR / f"{name}_sobel.jpg"), sobel)
    cv.imwrite(str(EDGES_DIR / f"{name}_canny.jpg"), canny)
    cv.imwrite(str(EDGES_DIR / f"{name}_laplacian.jpg"), laplacian)

    input_images[f"{name}_sobel"] = sobel
    input_images[f"{name}_canny"] = canny
    input_images[f"{name}_laplacian"] = laplacian

for name, input_image in input_images.items():
    boxed = box_dice(image, input_image)
    circled = circle_pips(image, input_image)

    cv.imwrite(str(ANNOTATED_DIR / f"{name}_boxed.jpg"), boxed)
    cv.imwrite(str(ANNOTATED_DIR / f"{name}_circled.jpg"), circled)