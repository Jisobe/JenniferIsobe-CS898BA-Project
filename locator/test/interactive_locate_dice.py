import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import (
    interact, FloatSlider, IntSlider, Dropdown, fixed, Layout, ToggleButton
)

def contour_solidity(contour):
    area = cv.contourArea(contour)
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    if hull_area == 0:
        return 0
    return area / hull_area

def filter_dice_contours(contours, image_area, min_area_fraction, max_area_fraction,
                          min_aspect_ratio, max_aspect_ratio, min_solidity):
    min_area_px = min_area_fraction * image_area
    max_area_px = max_area_fraction * image_area
    boxes = []
    for c in contours:
        area = cv.contourArea(c)
        if area < min_area_px or area > max_area_px:
            continue
        x, y, w, h = cv.boundingRect(c)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
        if contour_solidity(c) < min_solidity:
            continue
        boxes.append((x, y, w, h))
    return boxes

def locate_dice_tunable(
    image_bgr,
    blur_k=5,
    sigma=0.5,
    clahe_clip=2.0,
    clahe_tile=8,
    channel="saturation",
    morph_k=5,
    morph_iterations=1,
    min_area_frac=0.0015,
    max_area_frac=0.10,
    min_aspect=0.45,
    max_aspect=2.2,
    min_solidity=0.75,
):
    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    blur_k = blur_k if blur_k % 2 == 1 else blur_k + 1  # kernel must be odd
    blurred = cv.GaussianBlur(image_bgr, (blur_k, blur_k), sigma)

    # clahe = cv.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    # if channel == "l":
    #     lab = cv.cvtColor(blurred, cv.COLOR_BGR2LAB)
    #     raw_channel, _, _ = cv.split(lab)
    # else:
    #     hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    #     _, raw_channel, _ = cv.split(hsv)
    # normalized_channel = clahe.apply(raw_channel)
    # normalized_channel = blurred.apply(raw_channel)

    # _, threshold = cv.threshold(normalized_channel, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, threshold = cv.threshold(blurred, 75, 255, cv.THRESH_BINARY_INV)

    morph_k = morph_k if morph_k % 2 == 1 else morph_k + 1
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_k, morph_k))
    opened = cv.morphologyEx(threshold, cv.MORPH_OPEN, kernel, iterations=morph_iterations)
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel, iterations=morph_iterations)

    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxes = filter_dice_contours(
        contours, image_area, min_area_frac, max_area_frac,
        min_aspect, max_aspect, min_solidity
    )

    debug = {
        "blurred": cv.cvtColor(blurred, cv.COLOR_BGR2RGB),
        # "normalized_channel": normalized_channel,
        "threshold": threshold,
        "closed": closed,
    }
    return boxes, debug

def draw_boxes(image_bgr, boxes):
    annotated = image_bgr.copy()
    for (x, y, w, h) in boxes:
        cv.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 6)
    return cv.cvtColor(annotated, cv.COLOR_BGR2RGB)

def launch_tuner(image_path, figsize=(16, 9)):
    image = cv.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    slider_layout = Layout(width="450px")

    def _run(blur_k, sigma, clahe_clip, clahe_tile, channel, morph_k,
              morph_iterations, min_area_frac, max_area_frac,
              min_aspect, max_aspect, min_solidity, show_final_only):

        boxes, debug = locate_dice_tunable(
            image, blur_k, sigma, clahe_clip, clahe_tile, channel,
            morph_k, morph_iterations, min_area_frac, max_area_frac,
            min_aspect, max_aspect, min_solidity,
        )
        annotated = draw_boxes(image, boxes)

        if show_final_only:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.imshow(annotated)
            ax.set_title(f"Detected {len(boxes)} candidate die region(s)")
            ax.axis("off")
        else:
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            stages = [
                ("original", cv.cvtColor(image, cv.COLOR_BGR2RGB), None),
                ("blurred", debug["blurred"], None),
                (f"{channel} + CLAHE", debug["normalized_channel"], "gray"),
                ("otsu threshold", debug["threshold"], "gray"),
                ("morph open+close", debug["closed"], "gray"),
                (f"final: {len(boxes)} box(es)", annotated, None),
            ]
            for ax, (title, img, cmap) in zip(axes.flat, stages):
                ax.imshow(img, cmap=cmap)
                ax.set_title(title, fontsize=10)
                ax.axis("off")
        plt.tight_layout()
        plt.show()

        print(
            "\n--- paste into locate_dice.py ---\n"
            f"BLUR_KERNEL = ({blur_k if blur_k % 2 else blur_k + 1}, "
            f"{blur_k if blur_k % 2 else blur_k + 1})\n"
            f"SIGMA = {sigma}\n"
            f"CLAHE_CLIP_LIMIT = {clahe_clip}\n"
            f"CLAHE_TILE_GRID = ({clahe_tile}, {clahe_tile})\n"
            f"MORPH_KERNEL_SIZE = ({morph_k if morph_k % 2 else morph_k + 1}, "
            f"{morph_k if morph_k % 2 else morph_k + 1})\n"
            f"MIN_AREA_FRACTION = {min_area_frac}\n"
            f"MAX_AREA_FRACTION = {max_area_frac}\n"
            f"MIN_ASPECT_RATIO = {min_aspect}\n"
            f"MAX_ASPECT_RATIO = {max_aspect}\n"
            f"MIN_SOLIDITY = {min_solidity}\n"
            f"CHANNEL = \"{channel}\"\n"
            f"# morph_iterations = {morph_iterations} (not currently a locate_dice.py constant --\n"
            f"# add one if you want to tune this too)"
        )

    interact(
        _run,
        blur_k=IntSlider(value=5, min=1, max=21, step=2, description="blur kernel", layout=slider_layout),
        sigma=FloatSlider(value=0.5, min=0.0, max=5.0, step=0.1, description="sigma", layout=slider_layout),
        clahe_clip=FloatSlider(value=2.0, min=0.5, max=8.0, step=0.1, description="CLAHE clip", layout=slider_layout),
        clahe_tile=IntSlider(value=8, min=2, max=32, step=1, description="CLAHE tile", layout=slider_layout),
        channel=Dropdown(options=["l", "saturation"], value="saturation", description="channel"),
        morph_k=IntSlider(value=5, min=1, max=21, step=2, description="morph kernel", layout=slider_layout),
        morph_iterations=IntSlider(value=1, min=1, max=5, step=1, description="morph iters", layout=slider_layout),
        min_area_frac=FloatSlider(value=0.0015, min=0.0001, max=0.02, step=0.0001, readout_format=".4f",
                                   description="min area frac", layout=slider_layout),
        max_area_frac=FloatSlider(value=0.10, min=0.02, max=0.40, step=0.01, description="max area frac", layout=slider_layout),
        min_aspect=FloatSlider(value=0.45, min=0.1, max=1.0, step=0.05, description="min aspect", layout=slider_layout),
        max_aspect=FloatSlider(value=2.2, min=1.0, max=4.0, step=0.1, description="max aspect", layout=slider_layout),
        min_solidity=FloatSlider(value=0.75, min=0.3, max=1.0, step=0.01, description="min solidity", layout=slider_layout),
        show_final_only=ToggleButton(value=False, description="final only"),
    )

if __name__ == "__main__":
    import sys
    print("This module is meant to be imported in a Jupyter notebook, e.g.:\n")
    print('    from tune_locate_dice import launch_tuner')
    print('    launch_tuner("data/raw/d6_ - 8.jpeg")\n')
    if len(sys.argv) > 1:
        print(f"(Note: running as a script won't show the interactive widgets — "
              f"open a notebook and pass '{sys.argv[1]}' to launch_tuner instead.)")