# Biological Motion Perception (BMP) dataset

# Three Types of Visual Stimuli

## RGB Videos 

We select 10 action classes from the [RGB+D 120](https://rose1.ntu.edu.sg/dataset/actionRecognition/) dataset and apply a 7-to-3 ratio for dividing the data into training and testing splits. 
The details of the training and testing split are provided in [train_RGB.csv](Protocols/train_RGB.csv) and [test_RGB.csv](Protocols/test_RGB.csv).

## Joint Videos 

We apply [Alphapose](https://github.com/MVIG-SJTU/AlphaPose) to identify human body joints in the test set of our RGB videos. 
The model trained on [Halpe 26 keypoints](https://github.com/Fang-Haoshu/Halpe-FullBody) is used to generate our Joint videos.
Note that skeletons between joints are not visible to humans or AI models (no limbs).

## Sequential Position Actor Videos (SP)

We also use [Halpe 26 keypoints](https://github.com/Fang-Haoshu/Halpe-FullBody) to generate them. In this case, light points are positioned randomly between joints rather than on the joints themselves.

# Five Properties

## Temporal order (TO)

Reversal: [Reversal.py](Dataset/Reversal.py)

Shuffle: [Shuffle.py](Dataset/Shuffle.py)

## Temporal resolution (TR)

4 Frames: [4Frames.py](Dataset/4Frames.py)

3 Frames: [3Frames.py](Dataset/3Frames.py)

## Amount of visual information (AVI)

We quantify the amount of visual information based on the number of light points in Joint videos. 
Specifically, we included conditions: 5, 6, 10, 14, 18, and 26 light points. 
For the specific positions of light points, please read our paper. 
Retain only the essential points from the detected set of 26 key points provided by the Alphapose model [Halpe 26 keypoints](https://github.com/Fang-Haoshu/Halpe-FullBody).

## Lifetime of visual information (LVI)

Modify the code in AlphaPose/alphapose/utils/vis.py in [Alphapose](https://github.com/MVIG-SJTU/AlphaPose) such that 

8-point SP: Eight dots were randomly positioned on the eight limb segments between joints, with one dot on each limb segment.

4-point SP: Place four points on four limbs, which were also randomly selected from the total of eight limbs.

1/2/4 Lifetime: each dot remains at the same limb position for 1/2/4 frames before reallocating.

## Invariance to camera views (ICV)

Frontal View:  [test_6JOINTS_viewpoint_0.csv](Protocols/test_6JOINTS_viewpoint_0.csv)

45 degree View:  [test_6JOINTS_viewpoint_45.csv](Protocols/test_6JOINTS_viewpoint_45.csv)

90 degree View:  [test_6JOINTS_viewpoint_90.csv](Protocols/test_6JOINTS_viewpoint_90.csv)






