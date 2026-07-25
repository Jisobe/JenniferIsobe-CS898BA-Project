import argparse
import cv2 as cv
import numpy as np
from pathlib import Path

BLUR_KERNEL = (5, 5)
SIGMA = .5
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
MORPH_KERNEL_SIZE = (5, 5)
MIN_AREA_FRACTION = 0.0015
MAX_AREA_FRACTION = 0.1
MIN_ASPECT_RATIO = 0.45
MAX_ASPECT_RATIO = 2.2
MIN_SOLIDITY = 0.75
VALID_CHANNELS = ("l", "saturation")
THRESHOLD_METHODS = (cv.THRESH_OTSU, cv.THRESH_TRIANGLE)
MERGE_IOU_THRESHOLD = 0.3
DARK_BACKGROUND_THRESHOLD = 128
MIN_EXPECTED_DICE = 1
PIP_DEDUPE_IOU_THRESHOLD = 0.2
BORDER_MARGIN_PX = 3
PIP_RADIUS_REFERENCE_HEIGHT = 700
MIN_PIP_RADIUS_PX = 20
MAX_PIP_RADIUS_PX = 30

CURRENT_DIR = Path.cwd()
OUTPUT_DIR = CURRENT_DIR / "runs/localization/updated/run_737"
IMAGE_DIR = CURRENT_DIR / "data/raw/d6_ - 737.jpeg"
DEBUG = True


def contour_solidity(contour):
    area = cv.contourArea(contour)
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    if hull_area == 0:
        return 0
    return area / hull_area


def filter_dice_contours(contours, image_area, image_shape):
    image_h, image_w = image_shape[:2]
    min_area_px = MIN_AREA_FRACTION * image_area
    max_area_px = MAX_AREA_FRACTION * image_area
    boxes = []
    for c in contours:
        area = cv.contourArea(c)
        if area < min_area_px or area > max_area_px:
            continue

        x, y, w, h = cv.boundingRect(c)
        touches_border = (
            x <= BORDER_MARGIN_PX or
            y <= BORDER_MARGIN_PX or
            x + w >= image_w - BORDER_MARGIN_PX or
            y + h >= image_h - BORDER_MARGIN_PX
        )
        if touches_border:
            continue

        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            continue

        if contour_solidity(c) < MIN_SOLIDITY:
            continue

        boxes.append((x, y, w, h))

    return boxes


def box_iou(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def merge_box_candidates(candidate_lists, iou_threshold=MERGE_IOU_THRESHOLD):
    all_boxes = [box for boxes in candidate_lists for box in boxes]
    if not all_boxes:
        return []

    merged = []
    used = [False] * len(all_boxes)
    for i, box in enumerate(all_boxes):
        if used[i]:
            continue
        group = [box]
        used[i] = True
        for j in range(i + 1, len(all_boxes)):
            if used[j]:
                continue
            if box_iou(box, all_boxes[j]) >= iou_threshold:
                group.append(all_boxes[j])
                used[j] = True
        xs = np.mean([b[0] for b in group])
        ys = np.mean([b[1] for b in group])
        ws = np.mean([b[2] for b in group])
        hs = np.mean([b[3] for b in group])
        merged.append(((int(xs), int(ys), int(ws), int(hs)), len(group)))

    return merged


def select_expected_boxes(merged_with_votes, expected_count):
    """When the exact number of dice in the photo is known (RollCall always
    knows this -- it's however many were rolled/kept-out this turn), use
    that count to pick the most trustworthy boxes rather than just taking
    whatever the ensemble happened to produce.

    - More ensemble members agreeing on a box (higher vote count) is treated
      as higher confidence.
    - Ties are broken by how close a box's area is to the median area of all
      candidates, since a stray high-vote sliver or a merged double-box will
      usually still look like an outlier in size.
    - If there are more candidates than expected, the lowest-confidence
      extras are dropped. If there are fewer, everything found is kept and
      the caller (locate_dice) is responsible for topping up via the
      pip-clustering fallback.
    """
    if not merged_with_votes:
        return []

    areas = [w * h for (_, _, w, h), _ in merged_with_votes]
    median_area = float(np.median(areas))

    def sort_key(item):
        (_, _, w, h), votes = item
        area_penalty = abs(w * h - median_area)
        return (-votes, area_penalty)

    ranked = sorted(merged_with_votes, key=sort_key)
    keep = ranked[:expected_count] if expected_count else ranked
    return [box for box, _ in keep]


def threshold_channel(normalized_channel, method):
    _, thresh_img = cv.threshold(normalized_channel, 0, 255, cv.THRESH_BINARY + method)
    return thresh_img


def boxes_from_threshold(thresh_img, image_area, image_shape):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    opened = cv.morphologyEx(thresh_img, cv.MORPH_OPEN, kernel, iterations=3)
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return filter_dice_contours(contours, image_area, image_shape), closed


def cluster_points(points, max_dist):
    """Single-linkage clustering: group points that are within max_dist of
    any other point in the same group. Used to turn a scatter of detected
    pip centers into per-die groupings without an external clustering dep."""
    n = len(points)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
            if dist <= max_dist:
                union(i, j)

    clusters = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(points[i])
    return list(clusters.values())


def boxes_from_pip_clusters(image_bgr, grey_img, existing_boxes=None, max_new_boxes=None):
    """Fallback/supplemental localization for near-zero body-contrast cases
    (e.g. white dice on a light/marble background), where silhouette-based
    contour detection has no edge to find but individual pips are still
    visible. Detects pips via HoughCircles, single-linkage clusters the
    centers into per-die groups, then derives a bounding box (with padding)
    per cluster.

    existing_boxes: boxes already accepted from the silhouette ensemble.
        Clusters that overlap one of these are assumed to be the same die
        and are skipped, so this can be used to top up a partial result
        rather than only as an all-or-nothing fallback.
    max_new_boxes: if given (e.g. expected_dice_count - len(existing_boxes)),
        only the highest-confidence new clusters (most pips detected) are
        returned, since RollCall knows exactly how many more dice it needs.
    """
    existing_boxes = existing_boxes or []
    image_h, image_w = grey_img.shape[:2]

    # BUG FIX: minRadius/maxRadius used to be fixed pixel values tuned for a
    # specific image size. On higher-resolution photos, that range is
    # smaller than actual noise/texture specks, so Hough would latch onto
    # those instead of real pips (producing tiny boxes on nothing). Scale
    # both bounds by how much taller this image is than the reference size.
    radius_scale = image_h / PIP_RADIUS_REFERENCE_HEIGHT
    min_radius = max(3, int(round(MIN_PIP_RADIUS_PX * radius_scale)))
    max_radius = max(min_radius + 1, int(round(MAX_PIP_RADIUS_PX * radius_scale)))

    circles = cv.HoughCircles(
        grey_img,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=image_h / 85,
        param1=200,
        param2=10,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        return [], []

    circles = np.uint16(np.around(circles))[0]
    centers = [(int(c[0]), int(c[1])) for c in circles]
    avg_radius = float(np.mean([c[2] for c in circles]))

    # A single die face's pips are close together; different dice are
    # farther apart. Approximate one die's footprint from the area filter
    # already tuned for contour-based boxing, and use that as the linking
    # distance so pips belonging to the same die cluster together.
    image_area = image_h * image_w
    est_die_side = np.sqrt(MAX_AREA_FRACTION * image_area)
    max_link_dist = max(est_die_side * 0.6, avg_radius * 4)

    clusters = cluster_points(centers, max_link_dist)

    candidates = []
    for cluster in clusters:
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        pad = int(avg_radius * 2.5)
        x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
        x2, y2 = min(image_w, max(xs) + pad), min(image_h, max(ys) + pad)
        box = (x1, y1, x2 - x1, y2 - y1)

        if any(box_iou(box, existing) >= PIP_DEDUPE_IOU_THRESHOLD for existing in existing_boxes):
            continue

        candidates.append((box, len(cluster)))

    # More pips found in a cluster is a weak but useful confidence signal --
    # a real die face has 1-6 pips clustered tightly; a couple of stray
    # Hough false positives from background texture usually don't.
    candidates.sort(key=lambda item: -item[1])

    if max_new_boxes is not None:
        candidates = candidates[:max(max_new_boxes, 0)]

    boxes = [box for box, _ in candidates]
    return boxes, centers


def locate_dice(image_bgr, expected_dice_count=None, debug=False):
    """Locate dice in a photo.

    expected_dice_count: how many dice are actually in this photo. RollCall
        always knows this -- it's whatever was rolled or kept out this turn
        (1-5) -- so passing it lets locate_dice make much more confident
        decisions than guessing from image evidence alone:
          - if the ensemble proposes MORE boxes than expected, keep only the
            highest-confidence ones instead of over-reporting dice;
          - if it proposes FEWER, use pip-clustering to find the rest of the
            expected count rather than only kicking in as an all-or-nothing
            fallback when the ensemble finds literally zero.
        If None, falls back to the old count-agnostic behavior (accept
        whatever the ensemble merges to, only fall back to pip-clustering
        below MIN_EXPECTED_DICE).
    """
    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    # BUG FIX: this used to blur `image_area` (an int, h*w) instead of the
    # actual image, which would throw as soon as GaussianBlur ran.
    blurred = cv.GaussianBlur(image_bgr, BLUR_KERNEL, SIGMA)

    grey = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    background_is_dark = float(np.median(grey)) < DARK_BACKGROUND_THRESHOLD

    clahe = cv.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)

    lab = cv.cvtColor(blurred, cv.COLOR_BGR2LAB)
    l_channel, _, _ = cv.split(lab)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    _, s_channel, _ = cv.split(hsv)

    channels = {
        "l": clahe.apply(l_channel),
        "saturation": clahe.apply(s_channel),
    }

    candidate_box_lists = []
    threshold_debug = {}
    for channel_name, normalized_channel in channels.items():
        for method in THRESHOLD_METHODS:
            method_name = "otsu" if method == cv.THRESH_OTSU else "triangle"
            thresh_img = threshold_channel(normalized_channel, method)
            boxes, closed = boxes_from_threshold(thresh_img, image_area, image_bgr.shape)
            candidate_box_lists.append(boxes)
            threshold_debug[f"{channel_name}_{method_name}"] = thresh_img
            threshold_debug[f"{channel_name}_{method_name}_closed"] = closed

    merged_with_votes = merge_box_candidates(candidate_box_lists)

    if expected_dice_count is not None:
        boxes = select_expected_boxes(merged_with_votes, expected_dice_count)
    else:
        boxes = [box for box, _ in merged_with_votes]

    used_pip_fallback = False
    pip_centers = []
    target_count = expected_dice_count if expected_dice_count is not None else MIN_EXPECTED_DICE
    if len(boxes) < target_count:
        # Either the silhouette ensemble found nothing (the near-zero
        # body-contrast case, e.g. light dice on a light background) or it
        # found fewer boxes than we know should be in the photo. Either way,
        # top up with pip-clustering rather than under-reporting dice.
        used_pip_fallback = True
        needed = None if expected_dice_count is None else expected_dice_count - len(boxes)
        new_boxes, pip_centers = boxes_from_pip_clusters(
            image_bgr, grey, existing_boxes=boxes, max_new_boxes=needed
        )
        boxes = boxes + new_boxes

    circled = image_bgr.copy()
    for cx, cy in pip_centers:
        cv.circle(circled, (cx, cy), 3, (0, 0, 255), -1)

    debug_images = None
    if debug:
        debug_images = {
            "blurred": blurred,
            "grey": grey,
            **threshold_debug,
            "circled": circled,
        }
        debug_images["_meta"] = {
            "background_is_dark": background_is_dark,
            "used_pip_fallback": used_pip_fallback,
            "expected_dice_count": expected_dice_count,
            "detected_dice_count": len(boxes),
        }

    return boxes, debug_images


def draw_boxes(image_bgr, boxes):
    annotated = image_bgr.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 12)
    return annotated


def main():
    parser = argparse.ArgumentParser(description="Locate dice in a photo.")
    parser.add_argument("--image", type=Path, default=IMAGE_DIR, help="Path to input image")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Directory for annotated/debug output")
    parser.add_argument("--dice-count", type=int, default=5,
                         help="Number of dice actually in the photo (1-5), if known")
    parser.add_argument("--no-debug", action="store_true", help="Skip writing debug stage images")
    args = parser.parse_args()

    image = cv.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    debug_enabled = DEBUG and not args.no_debug
    boxes, debug_images = locate_dice(image, expected_dice_count=args.dice_count, debug=debug_enabled)

    print(f"Detected {len(boxes)} candidate die region(s):")
    for i, box in enumerate(boxes):
        print(f"  die {i}: x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated = draw_boxes(image, boxes)
    cv.imwrite(str(out_dir / "annotated.jpg"), annotated)
    print(f"\nAnnotated image saved to {out_dir / 'annotated.jpg'}")

    if debug_enabled and debug_images:
        meta = debug_images.pop("_meta", {})
        for name, img in debug_images.items():
            cv.imwrite(str(out_dir / f"debug_{name}.jpg"), img)
        print(f"Debug stage images saved to {out_dir}/debug_*.jpg")
        if meta:
            print(f"  background_is_dark={meta.get('background_is_dark')}, "
                  f"used_pip_fallback={meta.get('used_pip_fallback')}, "
                  f"expected_dice_count={meta.get('expected_dice_count')}, "
                  f"detected_dice_count={meta.get('detected_dice_count')}")


if __name__ == "__main__":
    main()