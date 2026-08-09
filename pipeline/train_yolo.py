"""
RollCall - YOLO Fine-Tuning

Fine-tunes a YOLO detection model on the Roboflow-exported dice dataset
(data/labeled/data.yaml). The dataset's classes are already per-face
("1".."6") bounding boxes -- one box per die, labeled with its face value --
so a single fine-tuned model handles both localization and classification
in one pass. There's no separate crop/classify training step like the
retired two-stage pipeline needed.

Ultralytics owns the training loop, LR scheduling, and early stopping
internally, and writes its own run artifacts to runs/yolo/<run_name>/
automatically:
    weights/best.pt, weights/last.pt  - checkpoints
    results.csv, results.png          - per-epoch train/val loss + metric
                                         curves (same role train.py's
                                         matplotlib block served for the
                                         custom CNN -- these ARE the
                                         overfitting-analysis curves, no
                                         hand-rolled plotting needed)
    confusion_matrix.png, PR_curve.png, args.yaml (full hyperparameter record)

Note on early stopping: the custom CNN's training loop asserted
PATIENCE > 2 * scheduler.patience + scheduler.cooldown because it paired
manual early stopping with a separate ReduceLROnPlateau scheduler, and an
early-stopping patience shorter than that could fire before the scheduler
ever got a chance to lower the LR. Ultralytics does not expose a
plateau-based scheduler in the same way (it uses its own cosine/linear LR
schedule internally), so `patience` below is just "epochs with no mAP
improvement before stopping" with no separate scheduler to cross-check it
against.

Usage:
    uv run train_yolo.py
    uv run train_yolo.py --model yolo26s.pt --epochs 150 --batch 8 --imgsz 1280
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

CURRENT_DIR = Path.cwd()
DATA_YAML = CURRENT_DIR / "data/labeled/data.yaml"
RUN_DATA_DIR = CURRENT_DIR / "runs"
CURR_MODEL_CLASS_DIR = RUN_DATA_DIR / "yolo"  # Mirrors runs/custom/ from the retired CNN pipeline

# Hyperparameters
BASE_MODEL = "yolo26n.pt"  # Pretrained checkpoint to fine-tune from. n=nano (fastest,
                            # least accurate) through x=extra-large. Start small; only
                            # size up if accuracy is the bottleneck, not speed.
EPOCHS = 150                # High ceiling -- Ultralytics' own early stopping (patience)
                            # will typically stop well before this is reached.
BATCH_SIZE = 16             # Lower if GPU memory becomes an issue. Full-resolution
                            # photos use far more memory per image than the CNN's
                            # 64x64 crops did, so this may need to drop lower than
                            # train.py's BATCH_SIZE = 32.
IMG_SIZE = 960              # Training/inference resolution. Dice occupy a small
                            # fraction of a full iPhone photo, so this needs to be
                            # much larger than the CNN's 64px crop size or small dice
                            # will shrink past recognizability after resizing. Try
                            # 640 first if memory-constrained, 1280 if recall on
                            # small/far dice is still weak at 960.
PATIENCE = 30                # Epochs with no mAP improvement before Ultralytics stops early.
RUN_NAME = "run_01"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on the RollCall dice dataset")
    parser.add_argument(
        "--model", type=str, default=BASE_MODEL,
        help=f"Base checkpoint to fine-tune from (default: {BASE_MODEL})"
    )
    parser.add_argument(
        "--data", type=str, default=str(DATA_YAML),
        help=f"Path to Roboflow-exported data.yaml (default: {DATA_YAML})"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument(
        "--name", type=str, default=RUN_NAME,
        help="Run name -- artifacts saved to runs/yolo/<name>/"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  RollCall - YOLO Fine-Tuning")
    print("=" * 50)
    print(f"  Base model  : {args.model}")
    print(f"  Data        : {args.data}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch}")
    print(f"  Image size  : {args.imgsz}")
    print(f"  Patience    : {args.patience}")
    print(f"  Run name    : {args.name}")
    print("=" * 50)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {data_path}. "
            "Point --data at your Roboflow-exported data.yaml "
            "(e.g. data/labeled/data.yaml)."
        )

    model = YOLO(args.model)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        project=str(CURR_MODEL_CLASS_DIR),
        name=args.name,
        exist_ok=False,  # Fail loudly on a name collision rather than silently
                          # overwriting a prior run's artifacts.
    )

    best_weights = CURR_MODEL_CLASS_DIR / args.name / "weights" / "best.pt"
    run_dir = CURR_MODEL_CLASS_DIR / args.name

    print("\nTraining complete.")
    print(f"Best weights   : {best_weights}")
    print(f"Run artifacts  : {run_dir}  (results.png, confusion_matrix.png, args.yaml, ...)")

    # results.png above covers train/val only, same distinction
    # test_interface.py drew for the two-stage pipeline -- val informs
    # early-stopping decisions during training, so it's not a fully
    # unbiased number. Run one pass over the held-out test split for that.
    print("\nEvaluating on held-out test split...")
    metrics = model.val(data=str(data_path), split="test")
    print(f"Test-split mAP50-95 : {metrics.box.map:.4f}")
    print(f"Test-split mAP50    : {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()