# Biological Motion Perception (BMP) dataset

Example videos in the BMP dataset are available at [link](https://drive.google.com/file/d/13cb9WdeQiuqmVzVCFMVhkHEadAid-EJ0/view?usp=sharing).

# Three Types of Visual Stimuli

## RGB Videos 

Please download RGB videos from the official [RGB+D 120](https://rose1.ntu.edu.sg/dataset/actionRecognition/) website. We select 10 action classes from the [RGB+D 120](https://rose1.ntu.edu.sg/dataset/actionRecognition/) dataset and apply a 7-to-3 ratio for dividing the data into training and testing splits. 
The details of the training and testing split are provided in [train_RGB.csv](Protocols/train_RGB.csv) and [test_RGB.csv](Protocols/test_RGB.csv).

## Joint Videos and Sequential Position Actor (SP) Videos

We apply [Alphapose](https://github.com/MVIG-SJTU/AlphaPose) to generate our Joint and SP videos from the test set of our RGB videos. 
The model trained on [Halpe 26 keypoints](https://github.com/Fang-Haoshu/Halpe-FullBody) is employed for this generation process..

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

b) To ensure that skeleton connections between joints remain invisible to both humans and AI models, we remove the limbs by deleting the code below in AlphaPose/alphapose/utils/vis.py (line 491 to line 519).

```
# Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                bg = img.copy()

                X = (start_xy[0], end_xy[0])
                Y = (start_xy[1], end_xy[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                stickwidth = (kp_scores[start_p] + kp_scores[end_p]) + 1
                polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length/2), int(stickwidth)), int(angle), 0, 360, 1)
                if i < len(line_color):
                    if opt.tracking:
                        cv2.fillConvexPoly(bg, polygon, color)
                    else:
                        cv2.fillConvexPoly(bg, polygon, line_color[i])
                else:
                    cv2.line(bg, start_xy, end_xy, (255,255,255), 1)
                if n < len(p_color):
                    transparency = float(max(0, min(1, 0.5 * (kp_scores[start_p] + kp_scores[end_p])-0.1)))
                else:
                    transparency = float(max(0, min(1, (kp_scores[start_p] + kp_scores[end_p]))))

                #transparency = float(max(0, min(1, 0.5 * (kp_scores[start_p] + kp_scores[end_p])-0.1)))
                img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
```

c) If AlphaPose fails to detect key points in a specific RGB frame of the video, the frame should be deleted instead of retaining the original. This can be achieved by removing the following code in AlphaPose/scripts/demo_inference.py (line 231).

```
writer.save(None, None, None, None, None, orig_img, im_name)
```

### Joint Videos
To produce videos with a varying number of joints, we undertake the following two steps:

1. Select joint indices from the set of 26 body keypoints detected by [Alphapose](https://github.com/MVIG-SJTU/AlphaPose) (refer to the body keypoints illustration in [Halpe](https://github.com/Fang-Haoshu/Halpe-FullBody) for more details). For example, in the J-6P condition, we select the following Joint indices and add this code after the code on line 27 (DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX) in AlphaPose/alphapose/utils/vis.py.

```
KP_26_2_6 =[0,9,10,19,15,16]
```

2. Modify the code on lines 472-482 in AlphaPose/alphapose/utils/vis.py by replacing

```
    for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= vis_thres[n]:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x), int(cor_y))
            bg = img.copy()
            if n < len(p_color):
                if opt.tracking:
                    cv2.circle(bg, (int(cor_x), int(cor_y)), 2, color, -1)
                else:
                    cv2.circle(bg, (int(cor_x), int(cor_y)), 2, p_color[n], -1)
```
with
```
 for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= vis_thres[n]:
                continue
            if n in KP_26_2_6:
                cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
                part_line[n] = (int(cor_x), int(cor_y))
                bg = img.copy()
                if n < len(p_color):
                    if opt.tracking:
                        cv2.circle(bg, (int(cor_x), int(cor_y)), 5, (255,255,255), -1)
                    else:
                        cv2.circle(bg, (int(cor_x), int(cor_y)), 5, (255,255,255), -1)
```

### SP Videos 

In SP videos, light points are positioned randomly between joints rather than on the joints themselves. To produce SP videos, we undertake the following two steps:






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






