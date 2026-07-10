# Research Notes

## Dice Detection using OpenCV

- Author: Gideon Vos
- Date: 9/14/2018
- [Source](https://gideonvos.wordpress.com/2018/09/17/dice-detection-using-opencv/)

Trying to detect color of dice and number of pips from video in real time

Using a YOLO/CNN model requires a high volume of images for training. Initial training with 40 images gave poor results. OpenCV is fast and doesn't require the training but it is limited to specific location, rotation, color and distance. HSV color mask to separate dice from background, detect dice using contours, detect pips with OpenCV HoughCircles.

## dice_recognizer

- Author: FaxabaduHacks
- Date: 6/2/2024
- [Source](https://github.com/FaxanaduHacks/dice_recognizer)

Find and detect dice faces from video feed.

Image binarization, contour detection, circularity analysis, and voting. Grey scales then Gaussian blur. Tested on dice with white pips and dominos with black pips. Dynamic settings/tresholding available to account for lighting variations. preprocessing, Binarization, and contour used to detect dice faces. Circularity analysis used for pips. Voting mechanism allows for consideration of multiple frames

## Dice-Detection

- Author: Kishaan
- Date: 6/15/2023
- [Source](https://github.com/Kishaan/Dice-Detection)

Detect dice location and number of pips on dice face from video. Dice thrown in try on a table. Considers hand movement, different lighting, noise, and clear/transparent dice. Uses OpenCV and SciPy hierarchical clustering. Uses 2 frames per second of video to reduce memory and time cost. Images converted to greyscale, absdiff, gaussian blur, and binary thresholding. Correct frame is selected by comparing non-zero pixels from one frame to the next and selecting the frame with the least movement. Bilateral filtering used to remove camera noise. Pips detected using openCV blobdetection. Hierarchical clustering is used to determine location of dice based on clustering of pips. Accuracy affected by video quality and camera angle.

Grey clustering could be looked at. Bounding each die. Bound the rolling surface

## dice-recognition

- Author: N3VERS4YDIE
- Date: 7/20/2025
- [Source](https://github.com/N3VERS4YDIE/dice-recognition)

Recognize dice, count and sum values in realtime from webcam/video input. Uses YOLO and OpenCV. Possible better performance with CUDA. Has dice dataset that can be used for training.

Little documentation on design process, success/failure metrics.

## dice_detection

- Author: nell-byler
- Date: 2/21/2022
- [Source](https://github.com/nell-byler/dice_detection)

***Very good resource***

Detect and classify dice from livestream feed. Classification is simpler than detection. Trained with 250 dice roll images. Labeled with labelImg. May run into some limitations with local training due to lack of GPU but can consider cloud computing. Incorporates model with mobile device using Android and TFLite

## Leveraging assistive technology for visually impaired people through optimal deep transfer learning based object detection model

- Authors: Mahir Mohammed Sharif Adam, Nojood O Aljehane, Mohammed Yahya Alzahrani, Samah Al Zanin
- Date: 8/17/2025
- [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC12358610/)

Focuses on image processing and deep learning techniques for visually impaired individuals.

## Computer vision for assistive technologies

- Authors: M. Leo, G. Medioni, M. Trivedi, T. Kanade, G.M. Farinella
- Date: 1/2017
- [Source](https://www.sciencedirect.com/science/article/abs/pii/S1077314216301357)

Reviews Computer Vision approaches to assistive technologies and points out improvements and impacts

## Computer vision based reliability control for electromechanical dice gambling machine

- Authors: I. Lapanja, M. Mraz, N. Zimic
- Date: 1/22/2000
- [Source](https://ieeexplore.ieee.org/document/854173/authors#authors)

Maybe out of date

Discusses dice location and number detection. Seems to rely on color difference for detection.

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
[dice-recognition github](https://github.com/N3VERS4YDIE/dice-recognition)
[D&D Dice Detection Computer Vision Dataset](https://universe.roboflow.com/thomas-phillips-t0qi6/d-d-dice-detection)
[D&D d6 Dice Computer Vision Model](https://universe.roboflow.com/andre-arante-cfckt/d-d-d6-dice)

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
