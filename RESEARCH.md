# Research Notes

## Dice Detection using OpenCV

- Author: Gideon Vos
- Date: 9/14/2018
- [Source](https://gideonvos.wordpress.com/2018/09/17/dice-detection-using-opencv/)
- Goal: Detect color of dice and number of pips from video in real time

Using a YOLO/CNN model requires a high volume of images for training (600+). Initial training with 40 images gave poor results. Has specific requirements including dice location, rotation, color and distance from dice to camera being within a specific, fixed range. OpenCV is fast and doesn't require the training but it is limited to specific location, rotation, color and distance. HSV color mask to separate dice from background, detect dice using contours, detect pips with OpenCV HoughCircles.

Likely not very useful for RollCall as requiring the specific needed for it to work is not very likely to happen. The live video feed addition would be interesting but such a setup would likely distract from the game/group dynamic.

## dice_recognizer

- Author: FaxabaduHacks
- Date: 6/2/2024
- [Source](https://github.com/FaxanaduHacks/dice_recognizer)
- Goal: Find and detect dice faces from live video feed.

Image binarization, contour detection, circularity analysis, and voting. Grey scales then Gaussian blur. Tested on dice with white pips and dominos with black pips. Dynamic settings/tresholding available to account for lighting variations. preprocessing, Binarization, and contour used to detect dice faces. Circularity analysis used for pips. Voting mechanism allows for consideration of multiple frames. Adjusters available to account for various lighting conditions.

Voting mechanism would be good to look into to increase confidence. Manual adjusting to account for different conditions is not reasonable for the RollCall application. Again video feed is an intersting approach but seems like not the best option for RollCall

## Dice-Detection

- Author: Kishaan
- Date: 6/15/2023
- [Source](https://github.com/Kishaan/Dice-Detection)
- Goal: Detect dice location and number of pips on dice face from video.

Dice thrown in tray on a table. Considers hand movement, different lighting, noise, and clear/transparent dice. Uses OpenCV and SciPy hierarchical clustering. Determining the correct frame to use for the actual dice detection can be difficult. Uses 2 frames per second of video to reduce memory and time cost. Images converted to greyscale, absdiff, gaussian blur (3x3 kernel), and binary thresholding. Correct frame is selected by comparing non-zero pixels from one frame to the next and selecting the frame with the least movement. Bilateral filtering used to remove camera noise (better detection of distinct features). The chosen frame is also cropped down to reduce the amount of pixels that need to be processed. Pips detected using openCV blobdetection. Hierarchical clustering is used to determine location of dice based on clustering of pips. Accuracy affected by video quality and camera angle. Dice sizes are predetermined have to fit within the min and max areas. The images are also assumed to not be at much of an angel (pips are graded based on similarity to circle shapes).

Grey clustering could be looked at. Bounding each die. Bound the rolling surface. The ability to account for transparent dice should be implemented/considered. Cutting down the image borders would be hard without having a fixed position camera. This is a very restrictive approach as far as camera positioning goes.

## dice-recognition

- Author: N3VERS4YDIE
- Date: 7/20/2025
- [Source](https://github.com/N3VERS4YDIE/dice-recognition)
- Goal: Recognize dice, count and sum values in realtime from webcam/video input.

Uses YOLOv8n.pt and OpenCV. Possible better performance with CUDA. Has large dice dataset that can be used for training.

Little documentation on design process or parameter tuning but there is a lot of really good output data and images like F-1 curve, Precision recall curve, Precision Confidence curve, recall confidence curve, confusion matrix, confusion matrix normalized. This could be good to use for checking the custom model against a pretrained yolo model

Dice dataset: <https://universe.roboflow.com/rune-zkmvl/rune55>

## dice_detection

- Author: nell-byler
- Date: 2/21/2022
- [Source](https://github.com/nell-byler/dice_detection)
- Goal: "Use deep learning to detect and classify six-sided dice from images, mobile devices, and (eventually) video"

***Very good resource***

Detect and classify dice from livestream feed. Classification is simpler than detection. Trained with 250 dice roll images. Labeled with labelImg. May run into some limitations with local training due to lack of GPU but can consider cloud computing. Incorporates model with mobile device using Android and TFLite. Need to be aware of what pre-trained model is being used to make sure it is compatible with mobile if wanting mobile. Docker is used to make it easier to deploy/integrate with a cloud environment (AWS). Batch sizing may become an issue with memory usage. Decrease batch size if not enough memory is available. Resizing image could also help with computation (64x64)

Provides information on deploying to AWS and GCS as well as mobile which could be useful later in the project. Mostly uses TensorFlow and Keras with a pre-trained model. Has a script for converting YOLO format to TFRecord format

YOLO is 0 indexed. Uses ReLU6, cosine_decay_learning_rate, batch_size 64, num_classes 6

cosine_decay_learning_rate {
  learning_rate_base: .04
  total_steps: 25000
  warmup_learning_rate: .013333
  warmup_steps: 2000
}

## TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

- Author: EdjeElectronics
- Date: 2020
- [Source](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#8-use-your-newly-trained-object-detection-classifier)

## Leveraging assistive technology for visually impaired people through optimal deep transfer learning based object detection model

- Authors: Mahir Mohammed Sharif Adam, Nojood O Aljehane, Mohammed Yahya Alzahrani, Samah Al Zanin
- Date: 8/17/2025
- [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC12358610/)
- - Goal: "Develop an effective object detection model for visually impaired individuals by utilizing advanced DL techniques"

Focuses on image processing and deep learning techniques for visually impaired individuals. Enhanced assistive Technology for Blind People through Object Detection Using a Hiking optimization algorithm (EATBP-ODHOA). Uses adaptive bilateral filtering (ABF) to reduce noise in the pre-processing stage and Faster R-CNN for object detection. Parameter tuning uses Hiking optimization algorithms (HOA). ABF reduces noise while preserving edges. Could be better than Gaussian. Uses region proposal network  with Faster R-CNN to reduce computational cost and allow for end to end training. RPN creates the bounding boxes and R-CNN classifies the objects. ResNet addresses vanishing gradient problem and DenseNet gives better generalization and reuse.

## Computer vision for assistive technologies

- Authors: M. Leo, G. Medioni, M. Trivedi, T. Kanade, G.M. Farinella
- Date: 1/2017
- [Source](https://www.sciencedirect.com/science/article/abs/pii/S1077314216301357)
- Goal: Create a task oriented categorization technique for assistive technology

This goes into way that assistive technology solutions and approaches can be categorized so that they can be used in other ways. It looks at Computer vision and object detection specifically. The need for is seems to be finding ways that assistive technology can be used aside from its original intended purpose that maybe was not considered initially.

## Computer vision based reliability control for electromechanical dice gambling machine

- Authors: I. Lapanja, M. Mraz, N. Zimic
- Date: 1/22/2000
- [Source](https://ieeexplore.ieee.org/document/854173/authors#authors)
- Reliability control module for a dice gambling machine (should produce readout in less the 1.5 sec)

***Maybe out of date***

Discusses dice location and number detection. Uses color chroma-keying: RGB is converted to HSV then cartisean values.

## An Auto-Recognizing System for Dice Games Using a Modified Unsupervised Grey Clustering Algorithm

- Author: Kuo-Yi Huang
- Date: 2/21/2008
- [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC3927534/)

Maybe out of date

Discusses recognizing score of dice using MUGCA. Identifies location and pips

## Dice Recognition in Uncontrolled Illumination Conditions by Local Invariant Features

- Authors: Gee-Sern Hsu, Hsiao-Chia Peng, Chyi-Yeu Lin, Pendry Alexandra
- Date: 2011
- [Source](https://link.springer.com/chapter/10.1007/978-3-642-23678-5_21)

Uses multiple cameras to combat poor lighting and uses MSER detector.

## Other Computer Vision applications for assistive technology

- [Assistive Technology for Seniors with Low Vision: Essential Daily Living Solutions That Restore Independence](https://nelowvision.com/assistive-technology-for-seniors-with-low-vision-essential-daily-living-solutions-that-restore-independence/)
- [Advancements in Assistive Technology for Low Vision](https://nelowvision.com/advancements-in-assistive-technology-for-low-vision/)

## Datasets

[Kaggle various Dice dataset](https://www.kaggle.com/datasets/ucffool/dice-d4-d6-d8-d10-d12-d20-images)
[dice-recognition github](https://universe.roboflow.com/rune-zkmvl/rune55)
[D&D Dice Detection Computer Vision Dataset](https://universe.roboflow.com/thomas-phillips-t0qi6/d-d-dice-detection)
[D&D d6 Dice Computer Vision Model](https://universe.roboflow.com/andre-arante-cfckt/d-d-d6-dice)
[Kaggle d6 dice](https://www.kaggle.com/datasets/nellbyler/d6-dice)

## Key Considerations

- Camera angle
- Lighting
- Color of dice
- Background color
- Clustering of dice
- Pip shape
- Pip size
- Image quality
- Processing time
- Accuracy
- Distance from dice to device taking the image
- Pip contour
- Reflection/glare on dice face
- Reflection/glare on pips
- Reflection/glare on background
- Ignoring non-dice objects
- Necessary processing power/time for CNN
- Numbers instead of pips

## Look Into

- TensorFlow object detection
  - [Tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#8-use-your-newly-trained-object-detection-classifier)
- OpenCV
  - blobdetection
  - HoughCircles
  - image preprocessing
- YOLO/CNN
- <https://academic.oup.com/gerontologist/advance-article/doi/10.1093/geront/gnag055/8662825?guestAccessKey=>
- <https://arxiv.org/abs/2506.07830>
- <https://pmc.ncbi.nlm.nih.gov/articles/PMC9016506/>
- <https://pmc.ncbi.nlm.nih.gov/articles/PMC9869388/>

## Useful documentation

Ultralytics Yolo: <https://docs.ultralytics.com>
Pytorch torch.nn:
    - <https://docs.pytorch.org/docs/2.13/nn.html>
    - <https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html>
TorchVision: <https://docs.pytorch.org/vision/stable/index.html>
