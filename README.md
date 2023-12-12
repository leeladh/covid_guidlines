## Problem

The project take the video as an input and outputs whether the person in the video is follow the guidelines of the COVID-19 or not.
The guidelines are:
1:- Between two people there is enough distance.
2:- If the person is wearing a mask or not.

If the person break the guidelines then the app makes a "beep" sound.

#### The Approach

The project uses the 3 types of deep learning CNN model. The models are:
1:- YOLOV3 - Predicts the location of the person in the video frame.
2:- ResNet - Predicts the location of the face of the person in the video frame.
3:- Mask Classifier - Predicts whether the person is wearing the mask or not.

First, The YOLO model predicts coordinates of the person present in the video. The output format of the YOLO model is (center_x, center_y, width, height, scores_of_objects). I took only the person score in consideration as we need only person coordinates. After that if the Euclidean distance between the (center_x, center_y) of two person is less than a threshold value (taken as 140.0), then we print the "Alert" above the box of the person if not then we will show the "SAFE" above the box of the person which surrounds the person.

After that, the frame of the video is passed through the ResNet model and predict the location of the face of the person. The output format of the ResNet model is (confidence_score, x, y, width, height) where the "confidence_score" tells how much the model is confident that there is a face located in the box coordinated as ((x, y), (x+w, y+h)). If the "confidence_score" is above a threshold (taken as 0.3) then we will pass the cropped face image of the person through the "Mask Classifier" model which will output the probability of the "mask" and "not mask". If the probablity of the "mask" is high than "high mask" then we will print "Mask" above the box of that specific person otherwise vice-versa.

After that we will show that video in a frame and after the video end we will free all memory.
The prediction is calculated at 20 FPS.

#### Setup

To run the file create a conda environment as follows:
```
conda create -n tensorgo python=3.7
conda activate tensorgo
```

After activating the environment run the file as follows:
```
python3 app.py
```
