# TODO: Add other types of normalization and a CLI option to chose the normalization type

import argparse
import cv2 as cv
from pathlib import Path

YOLO_DATASET_DIR  = "data/labeled" # Directory where the YOLO dataset was exported from Roboflow
OUTPUT_DIR   = "data/cropped"      # Directory where cropped dice images are saved
PADDING_PX   = 8                   # Pixels of padding around each crop. Adjust to ensure edges are not cut off
OUTPUT_SIZE  = 64                  # Cropped image resize dimension in px. Results in an image of size OUTPUT_SIZE x OUTPUT_SIZE

# Maps labeled data directories to cropped image directories
SPLIT_MAP = {
    "train": "train",
    "valid": "validation",
    "test":  "test",
}

# Die face classes
CLASS_NAMES = ["1", "2", "3", "4", "5", "6"]

# Helper functions

def yolo_to_pixel(bound_label, width, height):
    center_x, center_y, box_width, box_height = bound_label
    x1 = int((center_x - box_width / 2) * width)
    y1 = int((center_y - box_height / 2) * height)
    x2 = int((center_x + box_width / 2) * width)
    y2 = int((center_y + box_height / 2) * height)
    return x1, y1, x2, y2

def add_padding(x1, y1, x2, y2, padding, width, height):
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    return x1, y1, x2, y2

def apply_clahe(image_bgr):
    lab = cv.cvtColor(image_bgr, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_normalized = clahe.apply(l)

    lab_normalized = cv.merge([l_normalized, a, b])
    return cv.cvtColor(lab_normalized, cv.COLOR_LAB2BGR)

def process_split(
        split_name,
        output_split_name,
        dataset_dir,
        output_dir,
        padding,
        output_size,
        apply_normalization):
    images_dir = Path(dataset_dir) / split_name / "images"
    labels_dir = Path(dataset_dir) / split_name / "labels"
    out_base    = Path(output_dir) / output_split_name
    counts = {}
    skipped = 0
    errors  = 0

    if not images_dir.exists():
        print(f"{images_dir} not found - skipping split '{split_name}'")
        return {}

    for cls in CLASS_NAMES:
        (out_base / cls).mkdir(parents=True, exist_ok=True)
        counts[cls] = 0

    image_files = sorted([
        file for file in images_dir.iterdir() if file.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])

    if not image_files:
        print(f"No images found in {images_dir}")
        return counts

    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")

        if not label_path.exists():
            skipped += 1
            continue

        image = cv.imread(str(img_path))
        if image is None:
            print(f"Could not read image: {img_path.name}")
            errors += 1
            continue

        image_height, image_width = image.shape[:2]

        with open(label_path, "r") as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]

        if not lines:
            skipped += 1
            continue

        for line_index, line in enumerate(lines):
            parts = line.split()

            if len(parts) != 5:
                print(f"Unexpected label format in {label_path.name} line {line_index + 1}:\n    '{line}' - skipping")
                continue

            try:
                class_index = int(parts[0])
                center_x, center_y, box_width, box_height = map(float, parts[1:])
            except ValueError:
                print(f"Could not parse line in {label_path.name}:\n    '{line}' - skipping")
                continue

            if class_index < 0 or class_index >= len(CLASS_NAMES):
                print(f"Unknown class index {class_index} in {label_path.name} - skipping")
                continue

            class_name = CLASS_NAMES[class_index]

            x1, y1, x2, y2 = yolo_to_pixel((center_x, center_y, box_width, box_height), image_width, image_height)
            x1, y1, x2, y2 = add_padding(x1, y1, x2, y2, padding, image_width, image_height)

            if x2 <= x1 or y2 <= y1:
                print(f"Invalid crop in {img_path.name} box {line_index}: ({x1},{y1},{x2},{y2}) - skipping")
                continue

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                print(f"Empty crop from {img_path.name} - skipping")
                continue

            if apply_normalization:
                normalized_crop = apply_clahe(crop)

            crop_resized = cv.resize(
                normalized_crop,
                (output_size, output_size),
                interpolation=cv.INTER_LINEAR
            )

            out_filename = f"{img_path.stem}_box{line_index:02d}.jpg"
            out_path = out_base / class_name / out_filename

            cv.imwrite(str(out_path), crop_resized, [cv.IMWRITE_JPEG_QUALITY, 95])
            counts[class_name] += 1

    return counts, skipped, errors

def print_summary(counts, skipped, errors):
    total = sum(counts.values())

    print(f"\n{'  Class':<8} {'  Crops':>6}")
    print(f"  {'-'*8} {'-'*6}")

    for cls in CLASS_NAMES:
        print(f"  {cls:<8} {counts.get(cls, 0):>6}")
    print(f"  {'-'*8} {'-'*6}")

    print(f"  {'TOTAL':<8} {total:>6}")
    if skipped > 0:
        print(f"{skipped} image(s) skipped (no label file found)")
    if errors > 0:
        print(f"{errors} image(s) had read errors")

    counts_list = [value for value in counts.values() if value > 0]
    if counts_list:
        max_count = max(counts_list)
        min_count = min(counts_list)
        if max_count > min_count * 2:
            print(f"\nClass imbalance detected: max={max_count}, min={min_count}.")
            print(f"  Balance classes by capturing more images of underrepresented faces or capping larger classes")

def main():
    # Build CLI parser options
    parser = argparse.ArgumentParser(
        description="Crop labeled dice from Roboflow YOLO dataset for CNN training"
    )
    parser.add_argument(
        "--dataset", type=str, default=YOLO_DATASET_DIR,
        help=f"Path to Roboflow export root (default: {YOLO_DATASET_DIR})"
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_DIR,
        help=f"Output directory for cropped images (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--padding", type=int, default=PADDING_PX,
        help=f"Pixels of padding to add around each crop (default: {PADDING_PX})"
    )
    parser.add_argument(
        "--size", type=int, default=OUTPUT_SIZE,
        help=f"Output crop size in pixels - square (default: {OUTPUT_SIZE})"
    )
    parser.add_argument(
        "--no-clahe", action="store_true",
        help="Skip CLAHE lighting normalization (applied by default)"
    )
    args = parser.parse_args()

    apply_normalization = not args.no_clahe

    print("=" * 50)
    print("  RollCall - Dice Crop Script")
    print("=" * 50)
    print(f"  Dataset dir : {args.dataset}")
    print(f"  Output dir  : {args.output}")
    print(f"  Padding     : {args.padding}px")
    print(f"  Output size : {args.size}x{args.size}px")
    print(f"  Normalization (CLAHE)  : {'yes' if apply_normalization else 'no'}")
    print(f"  Classes     : {CLASS_NAMES}")
    print("=" * 50)

    for roboflow_split, output_split in SPLIT_MAP.items():
        print(f"\n-- Processing split: {roboflow_split} -> {output_split}")

        result = process_split(
            split_name=roboflow_split,
            output_split_name=output_split,
            dataset_dir=args.dataset,
            output_dir=args.output,
            padding=args.padding,
            output_size=args.size,
            apply_normalization=apply_normalization
        )

        if result:
            counts, skipped, errors = result
            print_summary(counts, skipped, errors)
    print("\n")
    print("=" * 50)
    print("  Script complete. Cropped dataset ready at:", args.output)
    print("=" * 50)

if __name__ == "__main__":
    main()
