# Biological Motion Perception (BMP) dataset

Example videos in the BMP dataset are available at [link](https://drive.google.com/file/d/13cb9WdeQiuqmVzVCFMVhkHEadAid-EJ0/view?usp=sharing).

# Three Types of Visual Stimuli

## RGB Videos 

Please download RGB videos from the official [RGB+D 120](https://rose1.ntu.edu.sg/dataset/actionRecognition/) website. We select 10 action classes from the [RGB+D 120](https://rose1.ntu.edu.sg/dataset/actionRecognition/) dataset and apply a 7-to-3 ratio for dividing the data into training and testing splits. 
The details of the training and testing split are provided in [train_RGB.csv](Protocols/train_RGB.csv) and [test_RGB.csv](Protocols/test_RGB.csv).

## Joint Videos and Sequential Position Actor (SP) Videos

We apply [Alphapose](https://github.com/MVIG-SJTU/AlphaPose) to generate our Joint and SP videos from the test set of our RGB videos. 
The model trained on [Halpe 26 keypoints](https://github.com/Fang-Haoshu/Halpe-FullBody) is employed for this generation process.

Before generating Joint and SP videos, we need to complete the following three steps:

a) To remove the RGB background from the videos, we perform the following two steps:

1. Add an argument in AlphaPose/scripts/demo_inference.py
```
parser.add_argument('--no_background', default=False, action='store_true')
```

2. Modify the code on line 260 in AlphaPose/scripts/demo_inference.py by replacing
```
writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
```
with
```
if args.no_background:
    black_img=np.zeros(orig_img.shape)
    writer.save(boxes, scores, ids, hm, cropped_boxes, black_img, im_name)
else:
    writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
```

b) If AlphaPose fails to detect key points in a specific RGB frame of the video, the frame should be deleted instead of retaining the original. This can be achieved by removing the following code in AlphaPose/scripts/demo_inference.py (line 231).

```
writer.save(None, None, None, None, None, orig_img, im_name)
```

### Joint Videos
To generate Joint videos, replace the file AlphaPose/alphapose/utils/vis.py in [Alphapose](https://github.com/MVIG-SJTU/AlphaPose) with [JOINTS/vis.py](AlphaPose/JOINTS/vis.py).

### SP Videos 

In SP videos, light points are positioned randomly between joints rather than on the joints themselves. To generate SP videos, replace the file AlphaPose/alphapose/utils/vis.py in [Alphapose](https://github.com/MVIG-SJTU/AlphaPose) with [SP/vis.py](AlphaPose/SP/vis.py). 

# Five Properties

## Temporal order (TO)

You can reverse the frame order in RGB/JOINTS/SP videos using [Reversal.py](Dataset/Reversal.py).

You can shuffle the frame order in RGB/JOINTS/SP videos using [Shuffle.py](Dataset/Shuffle.py)

## Temporal resolution (TR)

You can reduce the number of frames in RGB/JOINTS/SP videos to 4 frames using [4Frames.py](Dataset/4Frames.py)

You can reduce the number of frames in RGB/JOINTS/SP videos to 3 frames using [3Frames.py](Dataset/3Frames.py)

## Amount of visual information (AVI)

We quantify the amount of visual information based on the number of light points in Joint videos. 
Specifically, we included conditions: 5, 6, 10, 14, 18, and 26 light points. 
For the specific positions of light points, please read our paper. 

To generate videos with a varying number of joints, replace the list "KP_26_2_18" on line 486 of  [JOINTS/vis.py](AlphaPose/JOINTS/vis.py) with the corresponding list explained below:
*  KP_26 : 26 joints indices for J-26P;
*  KP_26_2_18 : 18 joints indices for J-18P;
*  KP_26_2_14 : 14 joints indices for J-14P;
*  KP_26_2_10 : 10 joints indices for J-10P;
*  KP_26_2_6 : 6 joints indices for J-6P;
*  KP_26_2_5 : 5 joints indices for J-5P.

## Lifetime of visual information (LVI)

When generating SP videos, change the variable "life_time" on line 487 of [SP/vis.py](AlphaPose/SP/vis.py) to control the lifetime of visual information.

## Invariance to camera views (ICV)

The videos in the J-6P condition can be divided into J-6P-0V, J-6P-45V, and J-6P-90V categories based on the three CSV files listed below:

Frontal View:  [test_6JOINTS_viewpoint_0.csv](Protocols/test_6JOINTS_viewpoint_0.csv)

45 degree View:  [test_6JOINTS_viewpoint_45.csv](Protocols/test_6JOINTS_viewpoint_45.csv)

90 degree View:  [test_6JOINTS_viewpoint_90.csv](Protocols/test_6JOINTS_viewpoint_90.csv)






