import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from PIL import Image
import matplotlib.pyplot as plt
from classifier.cnn_dice_detection_model import DiceClassifier

IMG_SIZE = 64
NUM_CLASSES = 6
CLASS_NAMES = ["1", "2", "3", "4", "5", "6"]
BATCH_SIZE = 64
NORM_MEAN = [0.5096, 0.4681, 0.3996]
NORM_STD = [0.2762, 0.2643, 0.2510]


class DiceDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.skipped_classes = []

        for label in range(1, 7):
            class_dir = os.path.join(root_dir, str(label))
            if not os.path.exists(class_dir):
                self.skipped_classes.append(str(label))
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), label - 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def build_test_transform():
    return v2.Compose(
        [
            v2.Resize((IMG_SIZE, IMG_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )

def load_model(checkpoint_path: Path, device: str) -> nn.Module:
    model = DiceClassifier(classes=NUM_CLASSES, dropout_rate=0.0)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def run_inference(model, loader, device):
    all_predictions = []
    all_labels = []
    all_confidences = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        confidence, prediction = probs.max(dim=1)
        all_predictions.extend(prediction.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_confidences.extend(confidence.cpu().tolist())

    return np.array(all_labels), np.array(all_predictions), np.array(all_confidences)

def confusion_matrix(labels, prediction, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, prediction):
        matrix[true_label, pred_label] += 1
    return matrix

def per_class_performance(matrix):
    num_classes = matrix.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    support = matrix.sum(axis=1)

    for c in range(num_classes):
        tp = matrix[c, c]
        fp = matrix[:, c].sum() - tp
        fn = matrix[c, :].sum() - tp
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = (
            2 * precision[c] * recall[c] / (precision[c] + recall[c])
            if (precision[c] + recall[c]) > 0
            else 0.0
        )

    return precision, recall, f1, support

def plot_confusion_matrices(matrix, class_names, out_path):
    matrix_normalized = matrix.astype(np.float64) / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, data, title, fmt in (
        (ax1, matrix, "Confusion Matrix (counts)", "d"),
        (ax2, matrix_normalized, "Confusion Matrix (row-normalized)", ".2f"),
    ):
        im = ax.imshow(data, matrixap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted face")
        ax.set_ylabel("True face")
        ax.set_title(title)

        thresh = data.max() / 2.0 if data.max() > 0 else 0.5
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                text = f"{value:{fmt}}"
                ax.text(
                    j, i, text,
                    ha="center", va="center",
                    color="white" if value > thresh else "black",
                    fontsize=9,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def write_classification_report(class_names, precision, recall, f1, support, out_path):
    lines = []
    header = f"{'class':<8}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, name in enumerate(class_names):
        lines.append(
            f"{name:<8}{precision[i]:>10.3f}{recall[i]:>10.3f}{f1[i]:>10.3f}{support[i]:>10d}"
        )
    lines.append("-" * len(header))
    macro_p, macro_r, macro_f1 = precision.mean(), recall.mean(), f1.mean()
    total_support = int(support.sum())
    lines.append(
        f"{'macro avg':<8}{macro_p:>10.3f}{macro_r:>10.3f}{macro_f1:>10.3f}{total_support:>10d}"
    )
    report = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(report + "\n")
    return report

def main():
    parser = argparse.ArgumentParser(description="Evaluate DiceClassifier on the test split data")
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to a pretrained model (e.g. runs/custom/run_19/best_model.pt)",
    )
    parser.add_argument(
        "--test-dir", type=Path, default=Path("data/cropped/test"),
        help="Directory of the test split (default: data/cropped/test)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Where to save results (default: <checkpoint's run dir>/test_results)",
    )
    parser.add_argument(
        "--low-confidence-threshold", type=float, default=0.60,
        help="Flag test predictions below this confidence (default: 0.60)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    output_dir = args.output or (args.checkpoint.parent / "test_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = DiceDataset(args.test_dir, transform=build_test_transform())
    if test_dataset.skipped_classes:
        print(
            f"WARNING: no folder found for class(es) {test_dataset.skipped_classes} "
            f"under {args.test_dir} -- these classes will have zero support "
            f"in the confusion matrix below, not an error."
        )
    if len(test_dataset) == 0:
        raise RuntimeError(f"No test images found under {args.test_dir}")

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")

    model = load_model(args.checkpoint, device)
    labels, prediction, confidences = run_inference(model, test_loader, device)

    overall_accuracy = float((labels == prediction).mean())
    matrix = confusion_matrix(labels, prediction, NUM_CLASSES)
    precision, recall, f1, support = per_class_performance(matrix)

    low_confidence_mask = confidences < args.low_confidence_threshold
    low_confidence_count = int(low_confidence_mask.sum())
    low_confidence_accuracy = (
        float((labels[low_confidence_mask] == prediction[low_confidence_mask]).mean())
        if low_confidence_count > 0 else None
    )

    print(f"\nOverall test accuracy: {overall_accuracy:.4f} ({int((labels == prediction).sum())}/{len(labels)})")
    print(
        f"Predictions below {args.low_confidence_threshold:.0%} confidence: {low_confidence_count}/{len(labels)} "
        + (f"(accuracy: {low_confidence_accuracy:.4f})" if low_confidence_accuracy is not None else "")
    )

    report_path = output_dir / "classification_report.txt"
    report_text = write_classification_report(CLASS_NAMES, precision, recall, f1, support, report_path)
    print("\n" + report_text)

    matrix_plot_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrices(matrix, CLASS_NAMES, matrix_plot_path)
    print(f"\nConfusion matrix saved to: {matrix_plot_path}")

    metrics = {
        "checkpoint": str(args.checkpoint),
        "test_dir": str(args.test_dir),
        "num_test_samples": len(test_dataset),
        "overall_accuracy": overall_accuracy,
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": int(support[i]),
            }
            for i in range(NUM_CLASSES)
        },
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "low_confidence_threshold": args.low_confidence_threshold,
        "low_confidence_count": low_confidence_count,
        "low_confidence_accuracy": low_confidence_accuracy,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics JSON saved to: {metrics_path}")

if __name__ == "__main__":
    main()