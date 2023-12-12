import numpy as np
import time
import cv2
import math
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

time_duration = 0.2
freq = 2550
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
weightsPath = "yolo-coco/yolov3.weights"
configPath = "yolo-coco/yolov3.cfg"

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
confidence_threshold = 0.2

print("[INFO] loading face detector model...")
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

model_store_dir= "classifier.model"
maskNet = load_model(model_store_dir)

cap = cv2.VideoCapture('video.mp4')
index = 0
cnt = 0
x_prev = y_prev = 0
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)
result = cv2.VideoWriter('filename.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, (800, 512))

while (cap.isOpened()):
    ret, image = cap.read()
    
    if ret == False:
        break
    index += 1

    if index%20 == 0:
        image = cv2.resize(image, (640, 360))
        (H, W) = image.shape[:2]
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        print("Time taken to predict the image: {:.3f}seconds".format(end-start))
        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.1 and classID == 0:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        a = []
        b = []
        c = []
        e = []

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                a.append(x)
                b.append(y)
                c.append(w)
                e.append(h)

        distance = []
        sd = []
        for i in range(0, len(a) - 1):
            for k in range(1, len(a)):
                if (k == i):
                    break
                else:
                    x_dist = ((a[k] + (c[k] / 2)) - (a[i] + (c[i] / 2)))
                    y_dist = ((b[k] + (e[k] / 2)) - (b[i] + (e[i] / 2)))
                    d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                    distance.append(d)
                    if (d <= 140.0):
                        sd.append(i)
                        sd.append(k)
                    sd = list(dict.fromkeys(sd))

        if len(idxs) > 0:
            boxes = [boxes[i] for i in idxs.flatten()]
            per = 0
            for i in range(len(boxes)):
                per += 1
                if (i in sd):
                    color = (0, 0, 255)
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    text = "Alert"
                    os.system('play -nq -t alsa synth {} sine {}'.format(time_duration, freq))
                else:
                    color = (138, 68, 38)
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    text = 'SAFE'
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(image, "id: " + str(per), (x, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (51, 0, 25), 1)



        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (416, 416), (104.0, 177.0, 123.0))

        faceNet.setInput(blob)
        detections = faceNet.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = image[startY:endY, startX:endX]
                if np.shape(face) == ():
                    break
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                (mask, without_mask) = maskNet.predict(face)[0]
                label = "Mask" if mask > without_mask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                label = "{}".format(label)
                cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 2)
                print("End of classifier")

        im = cv2.resize(image, (800, 512))
        result.write(im)
        cv2.imshow("frame", im)
        cnt = cnt + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.getWindowProperty('frame', 4) < 1:
            break

cap.release()
result.release()
cv2.destroyAllWindows()