# RollCall

RollCall is an playing aid for Yahtzee that assists with reading dice faces and determining valid combinations.

Yahtzee is a dice rolling game played with five six-sided dice over 13 rounds. The objective of the game is to get the highest cumulative score by balancing risk taking, probability, and scorecard management. Each player starts with a blank scorecard with an upper and lower section. The upper section contains a box for each number on the side of the dice. Scores are added to the box by totaling value of all of the dice for that number. The lower section contains three of a kind, four of a kind, full house, small straight, large straight, chance, and Yahtzee sections. During each turn, a player rolls their 5 dice and may choose to re-roll any of those 5 dice up to two more times. The player can at any time in their rolling chose to score their turn by adding the value to an appropriate available box.

This is an exciting yet approachable game with simple mechanics. However, for those struggle with reading the pips on the dice or the text on the score sheet or those who have trouble tracking combinations, the game becomes exceedingly more difficult. RollCall assists those with such struggles by scanning the faceup pips on the dice and displaying the dice faces in large images for the player. They can then choose the dice they want to keep which will be stored in the RollCall system and re-roll as many dice as necessary. RollCall assists with scoring by providing a large text score sheet for players to select from and calculating the total for that box. RollCall also optionally provides a list of possible scoring combinations that the showing dice on any given roll satisfies assisting those with cognitive or processing issues or those just learning the game. While fully digital version of this game do exists, it doesn't hold the same gathering together to play games and make memories that sitting around the table rolling the dice does.

## Process Overview

1. Player rolls the dice
2. Player takes a picture of the dice
3. RollCall identifies the dice faces and values
4. RollCall will identify valid dice combinations
5. Player will select dice to keep. If all dice are kept, step 8 is performed
6. RollCall will digitally set aside those dice and the player will re-roll the number of dice that were not kept.
7. Steps 1-6 are repeated up to two more times
8. The final combination is selected and scored in the appropriate box

## Scope

RollCall will process an image containing the faceup side of five standard six-sided dice from a given roll. It will identify the faceup value of each die and determine valid Yahtzee combinations. The value of the dice as well as the combinations will displayed for player review.

## Upgrades/Stretch Goals

While this project is part of a computer vision course and will focus on the computer vision aspects, I think there are other ways that games like Yahtzee could be made more accessible other than just making them completely online.

* Motion detection to auto-capture dice values on a roll
* Digital dice for those who cannot roll (or maybe a mechanism for rolling the physical dice for them like a conveyor belt and dice tower). The digital dice runs the risk of making it feel too online
  * Mechanism to separate the kept dice from re-roll dice
  * Mechanism to collect the dice for re-rolling

## Tools used

* iPhone 15 Pro Camera: Used in the photo setting to capture images of dice. Each image contains 5 dice in various lightings including bright, medium, and low. The lighting was also placed at different angles in reference to the dice to give different shadow and lighting effects. The camera was places at one of 3 angels: roughly 90 degrees (straight above the dice), 55 degree, and 35 degrees. Images were saved in the JPEG format and are located in *./data/raw*
* Roboflow: Used to label the images. Each dice in each image was bounded with the bounding tool and then label with a class with values 1, 2, 3, 4, 5, or 6 based on the face up value of the dice. After labeling the images, they were added to a dataset and divided into training, validation, and test sets using a 70-20-10 split. The dataset was then exported in the YOLOv8 format and download then saved under *./data/labeled*

## crop_script.py

Used to crop each labeled die from the yolo images and save the new cropped images for training.
YOLO images can be used directly when using the YOLO model, cropped images are used for custom model training.

### Running the script

```bash
uv run crop_script.py
```

TODO: Add options and descriptions
