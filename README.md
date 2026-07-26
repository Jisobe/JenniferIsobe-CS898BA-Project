# RollCall

## Table of Contents

- [Summary](#summary)
- [Process Overview (player)](#process-overview-player)
- [Scope](#scope)
- [Upgrades/Stretch Goals](#upgrades--stretch-goals)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
  - [Getting Data in Place](#getting-data-in-place)

## Summary

RollCall is a playing aid for Yahtzee that assists with reading dice faces and determining valid scoring combinations.

Yahtzee is a dice rolling game played with five six-sided dice over 13 rounds. The objective of the game is to get the highest cumulative score by balancing risk taking, probability, and scorecard management. Each player starts with a blank scorecard with an upper and lower section. The upper section contains a box for each number on the side of the dice. Scores are added to the box by totaling the value of all of the dice for that number. The lower section contains three of a kind, four of a kind, full house, small straight, large straight, chance, and Yahtzee sections. During each turn, a player rolls their 5 dice and may choose to re-roll any of those 5 dice up to two more times. The player can at any time in their rolling choose to score their turn by adding the value to an appropriate available box.

This is an exciting yet approachable game with simple mechanics. However, for those who struggle with reading the pips on the dice or the text on the score sheet, or those who have trouble tracking combinations, the game becomes exceedingly more difficult. RollCall assists those with such struggles by scanning the faceup pips on the dice and displaying the dice faces in large images for the player. They can then choose the dice they want to keep, which will be stored in the RollCall system, and re-roll as many dice as necessary. RollCall assists with scoring by providing a large-text score sheet for players to select from and calculating the total for that box. RollCall also computes every valid scoring combination for the showing dice on any given roll, assisting those with cognitive or processing issues or those just learning the game. While fully digital versions of this game do exist, they don't hold the same "gathering together to play games and make memories" feeling that sitting around the table rolling the dice does.

This repository is the semester project for CS 898BA (Computer Vision). It features a from-scratch custom CNN classifier (`DiceClassifier`) paired with a classical computer-vision localization stage.

## Process Overview (player)

1. Player rolls the dice
2. Player takes a picture of the dice
3. RollCall identifies the dice faces and values
4. RollCall identifies valid dice combinations
5. Player selects the dice to keep. If all dice are kept, step 8 is performed
6. RollCall digitally sets aside those dice and the player re-rolls the number of dice that were not kept
7. Steps 1-6 are repeated up to two more times
8. The final combination is selected and scored in the appropriate box

## Scope

RollCall processes an image containing the faceup side of 1-5 standard six-sided dice from a given roll. It identifies the faceup value of each die and determines valid Yahtzee combinations. The value of the dice, as well as the combinations, are displayed for player review. Inputs are photo-based (JPEG), not live video.

## Repository Structure

The code is organized into three small packages (`classifier/`, `locator/`, `pipeline/`) plus top-level scripts for training, tuning, and evaluation. This mirrors the two-stage design: a classical-CV locator stage that finds each die in a photo, and a CNN classification stage that reads each dies face value.

```code
JenniferIsobe-CS898BA-Project/
|- classifier/
|   |- test/
|   |   |- test_hyperparameters.py    # script for tuning and testing various hyperparameter configs
|   |   |- test.py                    # script for testing the trained CNN model
|   |- cnn_dice_detection_model.py    # DiceClassifier CNN architecture
|   |- compute_mean_std.py            # Script to calculate dataset-specific normalization stats for train.py
|   |- train.py                       # Trains DiceClassifier. Runs training and validation passes using provided data
|- data/
|   |- raw/                           # Captured JPEG photos (5 dice per photo) *Not included in repo due to space constraints
|   |- labeled/                       # Roboflow YOLOv8 export (train/valid/test, 70/20/10)
|   |- cropped/                       # Per-die crops sorted into class folders 1-6
|- locator/
|   |- test/
|   |   |- interactive_locate_dice.py # Jupyter based interactive verison of locate_dice.py to help with tuning parameters
|   |   |- locate_failure_test.py     # Supplemental testing script to visualize location testing failures
|   |   |- test_locate_dice.py        # Test script for locate_dice.py
|   |- crop_script.py                 # Roboflow YOLO labels -> cropped training images
|   |- locate_dice.py                 # Original single-channel locator (Milestone 2)
|   |- manual_locate.py               # Manual multi-channel/threshold comparison script
|   |- upgraded_locate_dice.py        # Multi channel voting mechanism locator script used for app
|- pipeline/
|   |- test/
|   |   |- test_interface.py          # Full Stage 1 vs Stage 2 "localization tax" integration test
|   |- interface.py                   # DiceInferencePipeline: locate -> crop -> classify
|   |- milestone_2.py                 # Milestone 2 CLI demo: photo -> annotated result
|   |- scoring.py                     # Yahtzee category scoring and tracking
|- runs/
|   |- custom/run_N/                  # DiceClassifier training and testing runs (weights, curves, metrics)
|   |- localization/                  # locate_dice tuning and test results
|   |- pipeline/                      # Output results from milestone 2
|   |- tuning/                        # test_hyperparameters.py results
|- AI_LOG.md                          # Log of AI interactions used for this project
|- app.py                             # Streamlit UI tying the pipeline and scorecard together
|- INSTRUCTIONS.md                    # Project instructions provided
|- RESEARCH.md                        # Literature review notes
```

## Setup

Dependencies are managed with **uv**, not pip.

```bash
uv sync
```

Core dependencies: `torch`, `torchvision`, `opencv-python`, `numpy`, `scipy` (for `cKDTree` pip clustering), `matplotlib`, `streamlit`, `Pillow`.

### Getting data in place

RollCall contains photos and trained checkpoints in the repo. However, if you want to start fresh:

1. Capture JPEG photos of 5 dice under varied lighting/angle/background into `data/raw/`.
2. Label each die's bounding box and face value (1-6) in [Roboflow](https://roboflow.com), export as **YOLOv8** format, and unzip into `data/labeled/` (expects `train/`, `valid/`, `test/` subfolders, each with `images/` and `labels/`).
3. Run the crop script (below) to turn those labels into per-die training crops.
4. Train the classifier.

## Usage

### 1. Crop training images from Roboflow labels

```bash
uv run locator/crop_script.py
```

Optional flags:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--dataset` | `data/labeled` | Roboflow export root |
| `--output` | `data/cropped` | Where cropped per-die images are written |
| `--padding` | `8` | Pixels of context kept around each labeled box |
| `--size` | `64` | Output crop size (square, px) - must match `IMG_SIZE` in `train.py` |
| `--no-clahe` | off | Skip CLAHE lighting normalization |

Output: `data/cropped/{train,validation,test}/{1..6}/*.jpg` and a printed per-class count table and a class-imbalance warning if any class has more than 2x the images of another.

### 2. Compute dataset-specific normalization stats

```bash
uv run compute_mean_std.py
```

Prints per-channel mean/std over `data/cropped/train`. Copy the printed `mean`/`std` into the `NORM_MEAN`/`NORM_STD` constants used by `train.py`, `test.py`, and `pipeline/interface.py` - these three **must stay in sync** or runtime normalization won't match what the model was trained on.

### 3. Train the classifier

```bash
uv run train.py
```

Edit the constants at the top of `train.py` to configure a run (learning rate, batch size, dropout, early-stopping patience, which `runs/custom/run_N` directory to write to). Outputs per run:

- `best_model.pt` - best-validation-loss checkpoint
- `hyperparameters.txt` - the config used, plus final best-epoch metrics
- `training_curves.png` - loss/accuracy curves (train vs. validation, required overfitting-analysis chart)
- `metrics.json` - full per-epoch loss/accuracy history

### 4. Run the hyperparameter test script

```bash
uv run test_hyperparameters.py --epochs 60 --configs baseline no_dropout no_augmentation
```

Runs several named configs (see table in [`test_hyperparameters.py`](#test_hyperparameterspy)) back-to-back and writes one overlaid plot (`runs/tuning/tuning_overlay.png`, solid = train, dashed = validation) so overfitting/underfitting behavior across configs can be compared on one chart. Omit `--configs` to run all of them.

### 5. Evaluate a trained checkpoint on the test split

```bash
uv run test.py --checkpoint runs/custom/run_20/best_model.pt
```

Reports overall test accuracy, a per-class precision/recall/F1 table, a confusion matrix, and flags low-confidence predictions (`--low-confidence-threshold`, default 0.60, matching `app.py`'s warning threshold).

### 6. Evaluate dice locator in isolation

```bash
uv run test_locate_dice.py --images-dir data/labeled/test/images
```

Reports detection recall, hallucination rate, mean IoU of matched boxes, and count-mismatch rate.

### 7. Run the full Stage 1 vs. Stage 2 integration test ("localization tax")

```bash
uv run test_interface.py --checkpoint runs/custom/run_20/best_model.pt --pip-mode fallback
```

The overall evaluation script - see [Localization Tax](#localization-tax) below for what it measures and why.

### 8. Diagnose the worst localization failures visually

```bash
uv run locator/locate_failure_test.py \
    --sweep-json runs/localization/test_results/localization_sweep.json \
    --images-dir data/labeled/test/images \
    --labels-dir data/labeled/test/labels \
    --top-n 10
```

Draws ground truth boxes (green) vs. predicted boxes (red - thicker if matched, thinner if hallucinated) on the worst-scoring images from a prior test_locate_dice.py run, and prints an undercounting vs. wrong-place diagnosis per image.

### 9. Run the end-to-end CLI demo

```bash
uv run milestone_2.py
```

Runs the full pipeline (locate -> crop -> classify) against a single hardcoded demo photo, saves an annotated JPEG with boxes and predicted face and confidence drawn on it, and prints the detected roll. This is the scripted, non-interactive version of what `app.py` does live. This script was used at the midpoint of development and has not be maintained to use updated logic. As such, this script should not be used for new runs.

### 10. Run the interactive app

```bash
uv run streamlit run app.py
```

Opens the full RollCall UI: upload a photo of your dice, see RollCall's read of each die (with a manual-override dropdown for any low-confidence or missed die), mark which dice to keep, re-roll the rest across up to 3 rolls, and record your score against a live Yahtzee scorecard (upper/lower sections, upper bonus, Yahtzee bonus).

## Data Pipeline

```code
Image Capture              Stage 1: Localization               Stage 2: Classification
──────────────             ──────────────────────              ────────────────────────
Player/dev photo   -->     locate_dice()                -->    DiceClassifier
(JPEG, 5 dice,              - Gaussian blur                     - crop resized to 64x64
 varied lighting/           - CLAHE (L and saturation           - CLAHE re-applied
 angle/background)            channels)                         - normalize (dataset mean/std)
                            - Otsu and Triangle threshold       - 3 conv blocks -> FC
                            - contour filtering
                            - voting mechanism
                            - pip-clustering fallback/vote
                                    \/
                           boxes (x, y, w, h) per die
                                    \/
                           crop and pad and CLAHE and resize (64x64)
                                    \/
                           DieDetection: (box, face 1-6, confidence)
                                    \/
                           app.py: digital dice faces and scorecard UI
```

### Development data flow (building the classifier)

```code
Roboflow label export (YOLOv8 box format, 70/20/10 train/valid/test split)
        \/
crop_script.py: yolo_to_pixel() -> add_padding() -> apply_clahe() -> resize(64x64)
        \/
data/cropped/{train,validation,test}/{1..6}/*.jpg
        \/
train.py: DiceDataset -> augment (train only) -> normalize -> DiceClassifier
        \/
runs/custom/run_N/{best_model.pt, training_curves.png, metrics.json}
        \/
test.py (test split)  and  test_interface.py (full-photo)
```

`test.py` measures the classifier against ground truth crops, while `test_interface.py` measures the classifier against `locate_dice()`'s actual crops on raw photos. The gap between them is the "localization tax".

## Code Explanations

### `classifier/cnn_dice_detection_model.py`

`DiceClassifier` is the custom CNN model skeleton for the classifier. It takes a 64x64x3 RGB crop of a single die and outputs 6 raw logits (one per face value):

- **3 convolutional blocks**, each `Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d(2,2)`, doubling channels each block: `3 -> 32 -> 64 -> 128`. Three 2x max-pools take the 64x64 input down to 8x8, so the flatten dimension is `128 * 8 * 8 = 8192`.
- **Classifier head**: `Flatten -> Linear(8192, 256) -> ReLU -> Dropout -> Linear(256, 64) -> ReLU -> Dropout -> Linear(64, 6)`.
- **No softmax in `forward()`** - `nn.CrossEntropyLoss` applies softmax internally during training, and every inference caller (`test.py`, `pipeline/interface.py`) applies `torch.softmax` explicitly on the raw logits when it needs a probability/confidence.
- `dropout_rate` is a constructor argument (default 0.5 in the class signature) so `test_hyperparameters.py` can test various configs without editing the model file; the best trained runs so far used 0.2-0.5 in practice.

### `locator/locate_dice.py`

The original, simpler localization approach from Milestone 2: blur -> CLAHE on a single channel (L or saturation) -> Otsu threshold -> morphological open/close -> external contours -> filter by area fraction, aspect ratio, and solidity. Kept in the repo as the documented baseline that `upgraded_locate_dice.py` evolved from.

### `locator/upgraded_locate_dice.py`

The current localization stage. Where the original used one channel and one threshold method, this runs a small ensemble:

1. Blur, then compute CLAHE-normalized L (Lab) and saturation (HSV) channels.
2. Threshold each channel with both Otsu and Triangle methods (4 candidate box sets total).
3. Filter each candidate set by area fraction, border-touching, aspect ratio, and contour solidity (same filters as the baseline, now border-aware).
4. Merge all candidate boxes across the 4 methods by IoU overlap, averaging boxes that agree and counting votes (`merge_box_candidates`).
5. If the expected dice count for this roll is known (RollCall should always know this - it's whatever was rolled/kept-out this turn), **`select_expected_boxes`** picks the highest-vote, most-typically-sized boxes rather than trusting raw ensemble output.
6. Pip-clustering fallback: if the ensemble still finds fewer boxes than expected, MSER-style pip detection (`HoughCircles`) and spatial clustering via `scipy.spatial.cKDTree` tops up the missing boxes. This is the path that seems most beneficial for cases like white dice on white marble, where the silhouette ensemble has nothing to threshold against. `--pip-mode always` runs this clustering unconditionally and lets pip-based candidates vote alongside the silhouette ensemble instead of only acting as a fallback. However, this comes at a much higher time and computation cost.
7. Optionally returns `candidate_groups` (the raw, un-averaged competing boxes behind each final box) so `pipeline/interface.py` can classify every competing crop for a die and keep whichever one the classifier is most confident about, instead of trusting the geometric average blindly.

A `StageTimer`/`print_timings` utility (`--time` flag) breaks down where the wall-clock cost goes (blur/color-convert, CLAHE, threshold and morphology, pip clustering, merge/select) for performance tuning.

### `locator/crop_script.py`

Converts a Roboflow YOLOv8 export into the per-die training images `train.py` consumes: reads each `images/*.jpg` and matching `labels/*.txt` pair, converts each YOLO-normalized `(cx, cy, w, h)` label to pixel coordinates (`yolo_to_pixel`), pads the box (`add_padding`), applies CLAHE on the L channel only (`apply_clahe` - Lab color space, not per-channel RGB, to avoid color distortion), resizes to a square, and writes it into `data/cropped/<split>/<class>/`. Reports a per-class crop count table and flags class imbalance (max more than 2x min).

### `locator/manual_locate.py` and `locator/interactive_locate_dice.py`

Development-only tuning tools, not part of the production pipeline. `manual_locate.py` batch-renders every channel/threshold/edge-detection combination for a set of known-hard reference photos (documented inline per photo: lighting, dice/background color pairing, and which combination worked). `interactive_locate_dice.py` exposes the same parameters as live Jupyter widgets (`ipywidgets`) for real-time visual tuning instead of edit-run-inspect cycles. These scripts were used to try to find the best combination of parameters for the locator that provided the best performance across all conditions.

### `train.py`

The training loop for `DiceClassifier`:

- **Augmentation** (train split only): `RandomRotation(15)` for rotation invariance, `ColorJitter(brightness=0.4, saturation=2)` for lighting variation. CLAHE (applied once at crop time in `crop_script.py`) is a preprocessing normalization.
- **Optimizer**: Adam, `weight_decay=1e-4`.
- **LR scheduler**: `ReduceLROnPlateau(patience=5, factor=0.5, cooldown=2)`.
- **Early stopping**: patience configurable (15 in the runs reported below), with a hard `assert PATIENCE > 2 * scheduler.patience and scheduler.cooldown` - this guarantees early stopping can never fire before the LR scheduler has had a chance to react to a plateau, which would otherwise stop training right as a LR drop was about to help.
- Saves the **deep-copied** best-validation-loss state dict, a `training_curves.png` (loss and accuracy, train vs. validation), and a full per-epoch `metrics.json`.

### `test_hyperparameters.py`

Runs a battery of named hyperparameter configs (baseline, no dropout, no weight decay, no augmentation, adjusted augmentation, and combinations) back-to-back on the same data, each with its own early stopping, and overlays every config's train/validation loss and accuracy curves on one pair of plots (`runs/tuning/tuning_overlay.png`) - solid lines are training, dashed are validation, one color per config. Useful for comparing performace of different configs.

### `compute_mean_std.py`

Streams every image in `data/cropped/train` through a resize-only transform and computes the true per-channel pixel mean/std for the dataset, replacing the ImageNet placeholder statistics that don't reflect this dataset's actual color distribution. `train.py`, `test.py`, and `pipeline/interface.py` all hardcode the printed result - re-run this whenever the training crop set changes meaningfully, and update all three files together.

### `test.py`

Final test-split evaluation for a trained checkpoint, deliberately kept separate from `train.py`'s validation loop. Produces `classification_report.txt` (per-class precision/recall/F1), `confusion_matrix.png` (raw and row-normalized), and `metrics.json`, and flags predictions below a confidence threshold.

### `pipeline/interface.py`

`DiceInferencePipeline` is the production glue between the locator and classification, used by both `app.py` and `test_interface.py`:

- `preprocess_to_tensor` does the resize/permute/normalize by hand instead of via `torchvision.transforms.Compose`/`ToPILImage`, to avoid a silent-failure mode.
- `process_photo()` calls `locate_dice()` with `return_candidates=True`, then for each die's group of *competing* candidate boxes, classifies every one and keeps whichever crop the model is most confident about (`_select_best_candidate`), letting the classifier resolve cases where the localization ensemble disagreed about the exact box for a die.
- `draw_annotated()` renders the "what RollCall saw" debug view used in the Streamlit sidebar/expander.

### `pipeline/scoring.py`

Pure Yahtzee rules logic. Takes a list of dice values in and returns scores:

- `score_category()` computes the score for one category given a dice roll (upper-section face totals, three/four of a kind, full house, small/large straight, Yahtzee, chance).
- `Scorecard` is a small stateful dataclass tracking which of the 13 categories are filled, computing upper subtotal/bonus, lower subtotal, and bonus-Yahtzee tracking (subsequent Yahtzees after the first score 100 bonus points each), and `grand_total()`.

### `app.py`

The Streamlit UI. Session state tracks the 5 dice slots (value and confidence and keep flag), rolls used this turn (max 3), and the running `Scorecard`. Flow per roll: upload a photo of only the dice being (re)rolled -> `DiceInferencePipeline.process_photo()` fills the open slots -> any slot RollCall couldn't confidently read gets an inline manual-override dropdown -> player marks keepers -> once all 5 slots are filled, every open scorecard category's score for the current roll is shown live so the player can pick where to record it. Falls back to fully manual dice entry if no model checkpoint is loaded, so the rest of the app (scoring, keep/reroll flow) can still be run without a trained model.

### `milestone_2.py`

A minimal, scripted (non-Streamlit) version of the same locate -> crop -> classify pipeline against one demo photo, used for midpoint testing and presenting. This script has not been maintained and should not be used.

### `test_locate_dice.py` and `locator/locate_failure_test.py`

`test_locate_dice.py` evaluates `locate_dice()` in isolation, reporting detection recall, hallucination rate, mean matched IoU, and count mismatch rate over a full labeled split. `locate_failure_test.py` consumes that run's JSON output and renders the worst-performing images with ground truth (green) vs. predicted (red) boxes overlaid, so a failure can be visually examined as "undercounting" vs. "confidently wrong place".

### `test_interface.py`

The full Stage 1 vs. Stage 2 integration test, see [Localization Tax](#localization-tax) below.

## Localization Tax

`test_interface.py` runs two evaluations over the same raw photos with ground truth YOLO labels, so the two numbers are directly comparable:

- **Stage 1 (classifier only)** - crop each die using its ground truth bounding box then classify.
- **Stage 2 (end-to-end)** - run the real `locate_dice()` on the raw photo (no ground truth boxes, only the expected dice count, exactly as `app.py` does), IoU-match predicted boxes to ground truth, classify each matched crop. A ground truth die that `locate_dice()` never finds counts as wrong.

```code
localization tax = Stage 1 accuracy − Stage 2 end-to-end accuracy
```

A tax near zero means `locate_dice()` isn't costing meaningful accuracy over a perfect localizer; a large tax means localization errors (misses, bad crops, hallucinations feeding the wrong die into a slot) are the most likely bottleneck.

## Current Results

Two representative training runs, both early-stopped

LR = `ReduceLROnPlateau(patience=5, factor=0.5, cooldown=2)`

early-stopping patience=15

| Run | LR | Batch | Dropout | Epochs | Best val loss | Val accuracy | Train accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `run_19` | 0.0005 | 64 | 0.2 | 200 | 0.2452 | 91.0% | 94.8% |
| `run_20` | 0.001 | 32 | 0.5 | 200 | 0.2107 | 94.0% | 85.3% |

`run_20`'s wider train/validation-accuracy gap in the other direction (train below validation) is expected here: dropout is active during training but disabled at eval time, combined with aggressive augmentation on the train split only.

Test-split classifier evaluation (`test.py`, `run_20` checkpoint) - macro F1 0.920:

| Face | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| 1 | 0.875 | 0.933 | 0.903 | 15 |
| 2 | 0.895 | 0.810 | 0.850 | 21 |
| 3 | 0.917 | 0.846 | 0.880 | 13 |
| 4 | 0.905 | 1.000 | 0.950 | 19 |
| 5 | 1.000 | 0.941 | 0.970 | 17 |
| 6 | 0.938 | 1.000 | 0.968 | 15 |

Full-pipeline integration test (`test_interface.py`, `run_20` checkpoint, `--pip-mode fallback`, 20 test photos / 100 ground truth dice):

| Metric | Value |
| --- | --- |
| Stage 1 accuracy | 93.0% |
| Detection recall | 63.0% |
| Hallucination rate | 25.0% |
| Mean matched IoU | 0.83 |
| Accuracy among matched dice | 82.5% |
| **End-to-end accuracy** | **52.0%** |
| **Localization tax** | **41 points** |

These results indicate that the classifier is performing well given a good crop image. But localization is the current bottleneck, with a 41-point end-to-end accuracy gap.

## Known Issues and Active Challenges

- While the addition of pip clustering on every run improves performance (especially when there is limited contrast between the dice and the background), the runtime and computational cost is too high. Multiple images took over 4 minutes to complete the pipeline, which for this application is not acceptable.
- The locator logic under-performs and limits the ability of the overall application.
- Low contrast (e.g., white dice on white marble) breaks the silhouette-based ensemble. Pip-clustering partially recovers these cases but is expensive to run.
- Touching/adjacent dice remain an outstanding edge case for both silhouette contours (they merge into one blob) and pip clustering (pips from two dice can cluster together).

## Next Steps

- Resolve or further document the touching dice localization failure case(s)
- Run a grayscale vs. RGB testing empirically
- Densify dataset labeling toward fuller Yahtzee-combination coverage before adding extra rolls of any single combination
- YOLO26 baseline comparison run

## Upgrades / Stretch Goals

While this project is part of a computer vision course and focuses on the computer vision aspects, there are other ways that games like Yahtzee could be made more accessible beyond making them completely online:

- Motion detection to auto-capture dice values on a roll
- Digital dice for those who cannot roll (or a mechanism for rolling the physical dice for them, like a conveyor belt and dice tower). Fully digital dice run the risk of making the experience feel too online
  - Mechanism to separate the kept dice from re-roll dice
  - Mechanism to collect the dice for re-rolling

## Tools Used

- **iPhone 15 Pro Camera**: used to capture `data/raw/` images. Each photo contains 5 dice under varied lighting (bright/medium/low), lighting angle, and camera angle (~90°, ~55°, ~35°), saved as JPEG.
- **Roboflow**: used to label each die's bounding box and face value (1-6), split 70/20/10 train/valid/test, and export in YOLOv8 format to `data/labeled/`.
- **uv**: Python package/dependency management (not pip).
- **PyTorch / torchvision**: `DiceClassifier` model, training loop, augmentation transforms.
- **OpenCV**: all classical-CV preprocessing and localization (CLAHE, thresholding, contours, `HoughCircles`).
- **SciPy** (`cKDTree`): spatial clustering for the pip-clustering localization fallback.
- **Streamlit**: the interactive `app.py` UI.

## Literature Review

See [`RESEARCH.md`](RESEARCH.md) for the full literature review notes and reference list informing both the localization design (Hsu et al.'s MSER pip-clustering precedent, Lapanja et al.'s classical chroma-keying approach, Huang's MUGCA touching-dice handling) and the classifier/pipeline split (nell-byler's two-stage detection/classification architecture and 64x64 input resolution precedent).
