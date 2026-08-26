# RollCall

RollCall is a playing aid for Yahtzee that assists with reading dice faces and determining valid combinations.

Yahtzee is a dice rolling game played with five six-sided dice over 13 rounds. The objective of the game is to get the highest cumulative score by balancing risk taking, probability, and scorecard management. Each player starts with a blank scorecard with an upper and lower section. The upper section contains a box for each number on the side of the dice. Scores are added to the box by totaling value of all of the dice for that number. The lower section contains three of a kind, four of a kind, full house, small straight, large straight, chance, and Yahtzee sections. During each turn, a player rolls their 5 dice and may choose to re-roll any of those 5 dice up to two more times. The player can at any time in their rolling choose to score their turn by adding the value to an appropriate available box.

This is an exciting yet approachable game with simple mechanics. However, for those who struggle with reading the pips on the dice or the text on the score sheet, or those who have trouble tracking combinations, the game becomes exceedingly more difficult. RollCall assists those with such struggles by scanning the faceup pips on the dice and displaying the dice faces in large images for the player. They can then choose the dice they want to keep, which will be stored in the RollCall system and re-roll as many dice as necessary. RollCall assists with scoring by providing a large text score sheet for players to select from and calculating the total for that box. RollCall also optionally provides a list of possible scoring combinations that the showing dice on any given roll satisfies, assisting those with cognitive or processing issues or those just learning the game. While fully digital versions of this game do exist, they don't hold the same gathering-together-to-play-games-and-make-memories feeling that sitting around the table rolling dice does.

## Process Overview

1. Player rolls the dice
2. Player takes a picture of the dice
3. RollCall identifies the dice faces and values
4. RollCall identifies valid dice combinations
5. Player selects dice to keep. If all dice are kept, step 8 is performed
6. RollCall digitally sets aside those dice and the player re-rolls the number of dice that were not kept
7. Steps 1-6 are repeated up to two more times
8. The final combination is selected and scored in the appropriate box

## Scope

RollCall processes an image containing the faceup side of five standard six-sided dice from a given roll. It identifies the faceup value of each die and determines valid Yahtzee combinations. The value of the dice, as well as the combinations, are displayed for player review.

## Architecture

RollCall detects and classifies dice in a single pass using a fine-tuned [YOLO26](https://docs.ultralytics.com/models/yolo26) model. One model finds each die in the photo *and* reads its face value at the same time, rather than locating dice with one method and classifying them with a separate model.

This is a deliberate change from an earlier two-stage approach (classical CV localization + a custom CNN classifier), which is preserved on `main` at the `milestone-3-final` tag for reference. That version's central finding — a 41-point accuracy gap between "given a perfect crop" (93%) and "found and read end-to-end" (52%) — showed localization, not classification, was the bottleneck. Switching to a single detector that learns both jobs jointly closed nearly all of that gap: the current pipeline reaches 100% detection recall, 0% hallucination rate, and 98% end-to-end accuracy on the held-out test split.

**Important implementation detail:** YOLO26 defaults to an NMS-free "one-to-one" inference head, which occasionally dropped a true detection or emitted a duplicate box on this dataset (small objects, several dice per image). `interface.py` explicitly passes `end2end=False` to `model.predict()`, switching to the traditional one-to-many head with real NMS post-processing. This trades a small, documented accuracy cost (~0.6-0.8 AP per Ultralytics' own benchmarks) for eliminating that failure mode entirely — see `docs/yolo_migration_report.md` for how this was diagnosed.

## Tools used

* **iPhone 15 Pro Camera**: used to capture dice images. Each image contains 5 dice under a deliberate factorial design of lighting (bright/medium/low), lighting angle, and camera angle (~90°, 55°, 35°). Images are JPEG, saved under `data/raw/`.
* **Roboflow**: used to label images. Each die was bounded with the bounding tool and labeled with a class (1-6) based on its faceup value. The dataset was split 70/20/10 (train/val/test) at the roll/combination level to prevent leakage, then exported in YOLOv8 format to `data/labeled/`.
* **Ultralytics (YOLO26)**: base model fine-tuned on the labeled dataset. See `train_yolo.py`.

## Scripts

### `train_yolo.py`

Fine-tunes a YOLO26 checkpoint on `data/labeled/data.yaml`.

```bash
uv run train_yolo.py --imgsz 960 --name run_01
```

See the script's docstring for the full hyperparameter list and defaults.

### `test_yolo.py`

End-to-end evaluation on the held-out test split, using the exact same inference path `app.py` uses. Reports detection recall, hallucination rate, mean matched IoU, classification accuracy, and end-to-end accuracy.

```bash
uv run test_yolo.py --weights runs/yolo/run_01/weights/best.pt
```

### `overlay_errors.py`

Draws ground-truth boxes (green) against predicted boxes (blue/red/orange) for any test image with a detection error, for visual debugging.

```bash
uv run overlay_errors.py --weights runs/yolo/run_01/weights/best.pt
```

### `debug_raw_predictions.py
`
Dumps every raw candidate box YOLO proposes for a single image, before and after NMS, with pairwise IoU between same-class boxes. Useful for diagnosing duplicate-detection or missed-detection cases.

```bash
uv run debug_raw_predictions.py --weights runs/yolo/run_01/weights/best.pt --image "data/labeled/test/images/<name>.jpeg"
```

### `app.py`

The Streamlit demo UI. Point it at your fine-tuned weights (default: `runs/yolo/rollcall/weights/best.pt`) and it will walk through the roll → detect → keep → re-roll → score flow described above.

```bash
uv run streamlit run app.py
```

## Current results

Evaluated on the held-out test split (20 images, 100 dice):

| Metric | Value |
| --- | --- |
| Detection recall | 100% |
| Hallucination rate | 0% |
| Mean matched IoU | 0.959 |
| Classification accuracy (matched dice) | 98% |
| End-to-end accuracy | 98% |

Full methodology and the migration story (including the YOLO26 NMS-free debugging investigation) are documented in `YOLO_REPORT.md`.
