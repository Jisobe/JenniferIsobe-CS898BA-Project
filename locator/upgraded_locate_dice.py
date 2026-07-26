import argparse
import time
from collections import OrderedDict
import cv2 as cv
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

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
PIP_CLUSTER_MAX_SPAN_MULTIPLIER = 1.5
PIP_MIN_DIST_RADIUS_MULTIPLIER = 1.5
CURRENT_DIR = Path.cwd()
OUTPUT_DIR = CURRENT_DIR / "runs/localization/updated/run_1"
IMAGE_DIR = CURRENT_DIR / "data/raw/d6_ - 1.jpeg"
DEBUG = True

class StageTimer:
    def __init__(self, store, name):
        self.store = store
        self.name = name

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._start
        self.store[self.name] = self.store.get(self.name, 0.0) + elapsed

def print_timings(timings, total_label="TOTAL"):
    total = sum(timings.values())
    name_width = max(len(k) for k in timings) if timings else 4
    print(f"\n--- locate_dice stage timings ({total_label}: {total*1000:.1f} ms) ---")
    for name, secs in timings.items():
        pct = (secs / total * 100) if total > 0 else 0
        print(f"  {name.ljust(name_width)}  {secs*1000:8.1f} ms  ({pct:5.1f}%)")
    print("---------------------------------------------------------\n")

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
        merged.append(((int(xs), int(ys), int(ws), int(hs)), len(group), group))

    return merged

def select_expected_boxes(merged_with_votes, expected_count):
    if not merged_with_votes:
        return []

    areas = [w * h for (_, _, w, h), _, _ in merged_with_votes]
    median_area = float(np.median(areas))

    def sort_key(item):
        (_, _, w, h), votes, _ = item
        area_penalty = abs(w * h - median_area)
        return (-votes, area_penalty)

    ranked = sorted(merged_with_votes, key=sort_key)
    keep = ranked[:expected_count] if expected_count else ranked
    return [(box, group) for box, _, group in keep]

def threshold_channel(normalized_channel, method):
    _, thresh_img = cv.threshold(normalized_channel, 0, 255, cv.THRESH_BINARY + method)
    return thresh_img

def boxes_from_threshold(thresh_img, image_area, image_shape):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    opened = cv.morphologyEx(thresh_img, cv.MORPH_OPEN, kernel, iterations=3)
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return filter_dice_contours(contours, image_area, image_shape), closed

def cluster_points(points, max_dist, max_cluster_span=None):
    n = len(points)
    parent = list(range(n))
    bounds = [[points[i][0], points[i][0], points[i][1], points[i][1]] for i in range(n)]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    links = []
    if n > 1:
        points_arr = np.asarray(points, dtype=np.float64)
        tree = cKDTree(points_arr)
        pairs = tree.query_pairs(r=max_dist, output_type="ndarray")
        if len(pairs) > 0:
            diffs = points_arr[pairs[:, 0]] - points_arr[pairs[:, 1]]
            dists = np.hypot(diffs[:, 0], diffs[:, 1])
            order = np.argsort(dists)
            links = [(float(dists[k]), int(pairs[k, 0]), int(pairs[k, 1])) for k in order]

    for _, i, j in links:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue

        if max_cluster_span is not None:
            bi, bj = bounds[ri], bounds[rj]
            merged_w = max(bi[1], bj[1]) - min(bi[0], bj[0])
            merged_h = max(bi[3], bj[3]) - min(bi[2], bj[2])
            if merged_w > max_cluster_span or merged_h > max_cluster_span:
                continue

        parent[rj] = ri
        bi, bj = bounds[ri], bounds[rj]
        bounds[ri] = [
            min(bi[0], bj[0]), max(bi[1], bj[1]),
            min(bi[2], bj[2]), max(bi[3], bj[3]),
        ]

    clusters = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(points[i])
    return list(clusters.values())


def boxes_from_pip_clusters(image_bgr, pip_search_img, existing_boxes=None, max_new_boxes=None, verbose=False,
                             timings=None, stage_prefix="pip"):

    existing_boxes = existing_boxes or []
    image_h, image_w = pip_search_img.shape[:2]
    own_timings = timings if timings is not None else OrderedDict()
    resize_scale = 1.0
    if image_h > PIP_RADIUS_REFERENCE_HEIGHT:
        resize_scale = PIP_RADIUS_REFERENCE_HEIGHT / image_h
        search_img = cv.resize(
            pip_search_img, None, fx=resize_scale, fy=resize_scale, interpolation=cv.INTER_AREA
        )
    else:
        search_img = pip_search_img

    circles = None
    with StageTimer(own_timings, f"{stage_prefix}:hough_circles"):
        circles = cv.HoughCircles(
            search_img,
            cv.HOUGH_GRADIENT,
            dp=1,
            minDist=MIN_PIP_RADIUS_PX * PIP_MIN_DIST_RADIUS_MULTIPLIER,
            param1=200,
            param2=10,
            minRadius=MIN_PIP_RADIUS_PX,
            maxRadius=MAX_PIP_RADIUS_PX
        )

    if circles is None:
        print("No circles")
        return [], []

    circles = np.uint16(np.around(circles))[0]
    centers = [(int(round(c[0] / resize_scale)), int(round(c[1] / resize_scale))) for c in circles]
    avg_radius = float(np.mean([c[2] for c in circles])) / resize_scale

    if verbose:
        print(f"  [pip-clustering] HoughCircles found {len(centers)} raw circle candidates")

    image_area = image_h * image_w
    est_die_side = np.sqrt(MAX_AREA_FRACTION * image_area)
    max_link_dist = max(est_die_side * 0.6, avg_radius * 4)

    with StageTimer(own_timings, f"{stage_prefix}:cluster_points"):
        clusters = cluster_points(centers, max_link_dist, max_cluster_span=est_die_side * PIP_CLUSTER_MAX_SPAN_MULTIPLIER)

    candidates = []
    rejected_oversized = 0
    for cluster in clusters:
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        pad = int(avg_radius * 2.5)
        x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
        x2, y2 = min(image_w, max(xs) + pad), min(image_h, max(ys) + pad)
        box_w, box_h = x2 - x1, y2 - y1
        box = (x1, y1, box_w, box_h)
        box_area_fraction = (box_w * box_h) / image_area if image_area else 0
        aspect_ratio = box_w / box_h if box_h > 0 else 0
        if (
            box_area_fraction < MIN_AREA_FRACTION
            or box_area_fraction > MAX_AREA_FRACTION
            or aspect_ratio < MIN_ASPECT_RATIO
            or aspect_ratio > MAX_ASPECT_RATIO
        ):
            rejected_oversized += 1
            continue

        if any(box_iou(box, existing) >= PIP_DEDUPE_IOU_THRESHOLD for existing in existing_boxes):
            continue

        candidates.append((box, len(cluster)))

    candidates.sort(key=lambda item: -item[1])

    if verbose and rejected_oversized > 0:
        print(
            f"  [pip-clustering] rejected {rejected_oversized} cluster(s) outside expected "
            f"die-footprint size/shape (likely single-linkage chaining across background "
            f"texture rather than a real die)"
        )

    if max_new_boxes is not None:
        candidates = candidates[:max(max_new_boxes, 0)]

    boxes = [box for box, _ in candidates]
    return boxes, centers


def locate_dice(image_bgr, expected_dice_count=None, debug=False, pip_mode="fallback", timings=None,
                 return_candidates=False):
    own_timings = timings if timings is not None else OrderedDict()
    image_area = image_bgr.shape[0] * image_bgr.shape[1]

    with StageTimer(own_timings, "blur_and_color_convert"):
        blurred = cv.GaussianBlur(image_bgr, BLUR_KERNEL, SIGMA)
        grey = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
        background_is_dark = float(np.median(grey)) < DARK_BACKGROUND_THRESHOLD
        lab = cv.cvtColor(blurred, cv.COLOR_BGR2LAB)
        l_channel, _, _ = cv.split(lab)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
        _, s_channel, _ = cv.split(hsv)

    with StageTimer(own_timings, "clahe"):
        clahe = cv.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
        channels = {
            "l": clahe.apply(l_channel),
            "saturation": clahe.apply(s_channel),
        }

    candidate_box_lists = []
    threshold_images = {}
    with StageTimer(own_timings, "threshold_and_morphology"):
        for channel_name, normalized_channel in channels.items():
            for method in THRESHOLD_METHODS:
                method_name = "otsu" if method == cv.THRESH_OTSU else "triangle"
                thresh_img = threshold_channel(normalized_channel, method)
                boxes, closed = boxes_from_threshold(thresh_img, image_area, image_bgr.shape)
                candidate_box_lists.append(boxes)
                threshold_images[f"{channel_name}_{method_name}"] = thresh_img
                threshold_images[f"{channel_name}_{method_name}_closed"] = closed

    pip_centers = []
    used_pip_as_parallel_check = False
    if pip_mode == "always":
        used_pip_as_parallel_check = True
        pip_boxes, pip_centers = boxes_from_pip_clusters(
            image_bgr, channels['l'], existing_boxes=[], max_new_boxes=None, verbose=debug,
            timings=own_timings, stage_prefix="pip_parallel"
        )
        candidate_box_lists.append(pip_boxes)

    with StageTimer(own_timings, "merge_and_select"):
        merged_with_votes = merge_box_candidates(candidate_box_lists)

        if expected_dice_count is not None:
            selected = select_expected_boxes(merged_with_votes, expected_dice_count)
        else:
            selected = [(box, group) for box, _, group in merged_with_votes]

        boxes = [box for box, _ in selected]
        candidate_groups = [group for _, group in selected]

    used_pip_fallback = False
    target_count = expected_dice_count if expected_dice_count is not None else MIN_EXPECTED_DICE
    if len(boxes) < target_count:
        used_pip_fallback = True
        needed = None if expected_dice_count is None else expected_dice_count - len(boxes)
        new_boxes, fallback_pip_centers = boxes_from_pip_clusters(
            image_bgr, channels['l'], existing_boxes=boxes, max_new_boxes=needed, verbose=debug,
            timings=own_timings, stage_prefix="pip_fallback"
        )
        boxes = boxes + new_boxes
        candidate_groups = candidate_groups + [[box] for box in new_boxes]
        if not pip_centers:
            pip_centers = fallback_pip_centers

    debug_images = None
    if debug:
        with StageTimer(own_timings, "debug_image_assembly"):
            circled = image_bgr.copy()
            for cx, cy in pip_centers:
                cv.circle(circled, (cx, cy), 3, (0, 0, 255), 5)

            debug_images = {
                "blurred": blurred,
                "grey": grey,
                "l_channel": channels["l"],
                "s_channel": channels["saturation"],
                **threshold_images,
                "circled": circled,
            }
            debug_images["_meta"] = {
                "background_is_dark": background_is_dark,
                "pip_mode": pip_mode,
                "used_pip_as_parallel_check": used_pip_as_parallel_check,
                "used_pip_fallback": used_pip_fallback,
                "expected_dice_count": expected_dice_count,
                "detected_dice_count": len(boxes),
            }

    if return_candidates:
        return boxes, debug_images, candidate_groups
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
    parser.add_argument("--pip-mode", choices=["fallback", "always"], default="fallback",
                         help="'fallback' (default): pip-clustering only runs to top up a "
                              "shortfall. 'always': pip-clustering runs unconditionally and "
                              "its candidates vote alongside the silhouette ensemble -- "
                              "helps touching-dice cases at the cost of extra compute.")
    parser.add_argument("--no-debug", action="store_true", help="Skip writing debug stage images")
    parser.add_argument("--time", action="store_true",
                         help="Print a per-stage timing breakdown for this run")
    args = parser.parse_args()

    image = cv.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    debug_enabled = DEBUG and not args.no_debug
    timings = OrderedDict() if args.time else None
    wall_start = time.perf_counter()
    boxes, debug_images = locate_dice(
        image, expected_dice_count=args.dice_count, debug=debug_enabled,
        pip_mode=args.pip_mode, timings=timings
    )
    wall_elapsed = time.perf_counter() - wall_start

    if args.time:
        print_timings(timings, total_label=f"instrumented sum, wall clock {wall_elapsed*1000:.1f} ms")

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
            print(f"  pip_mode={meta.get('pip_mode')}, "
                  f"background_is_dark={meta.get('background_is_dark')}, "
                  f"used_pip_as_parallel_check={meta.get('used_pip_as_parallel_check')}, "
                  f"used_pip_fallback={meta.get('used_pip_fallback')}, "
                  f"expected_dice_count={meta.get('expected_dice_count')}, "
                  f"detected_dice_count={meta.get('detected_dice_count')}")


if __name__ == "__main__":
    main()