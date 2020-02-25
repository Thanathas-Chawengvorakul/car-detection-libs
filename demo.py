import os
import cv2
import json
import numpy as np 
from time import time
import imutils
from Centroidtracker import CentroidTracker

"""
//? Capture Video
//? Scale down image and keep original size
//? use scaled image for detection and tracking
//? when show image use original size

//! generate config file from GUI
//? filter only some part
? when new id enter from opposite side -> Add it to FOCUS list
! capture car in focus list and save images in folder
"""


# REquired Files
PROTOTXT = 'models/model.prototxt'
CAFFE_WB = 'models/wb.caffemodel'
SOURCE = 'datasets/video3.mp4'
LOG_PATH = 'LOGS'
CONFIG = 'config.json'

#Required Variables
(newH, newW) = (600, 600)
CLASSES = np.array(["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"])
FOCUS_CLASSES = np.array(['car', 'motorbike'])
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
CONF = 0.5
with open('config.json', 'r') as jsonfile:
    config = json.load(jsonfile)
(x1, y1) = config['xY1']
(x2, y2) = config['xY2']
(x3, y3) = config['xY3']
(x4, y4) = config['xY4']
leftXy = (min(x1, x2), max(abs(y1), abs(y2)))
rightXy = (max(x3, x4), min(y3, y4))
(dX1, dY1) = config['danger1']
(dX2, dY2) = config['danger2']
dLeftXy = [dX2, dY1]
dRightXy = [dX1, dY2]
#crop = img[dRightXy[1]:dLeftXy[1], dRightXy[0]:dLeftXy[0]]

# Settings
print('Init Model and Camera...')
cam = cv2.VideoCapture(SOURCE)
net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFE_WB)
ct = CentroidTracker()
input('[Enter] to continue')

# Methods
def cropImage(img):
    global leftXy, rightXy
    return img[rightXy[1]:leftXy[1], leftXy[0]:rightXy[0]]

while 1:
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=1000)

    if not ret:
        break

    orig = frame.copy()
    (H, W) = frame.shape[:2]
    rH = H / float(newH)
    rW = W / float(newW)

    image = cv2.resize(frame, (newW, newH))
    (H, W) = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, 0.007843, (H, W), 127.5)
    net.setInput(blob)
    detections = net.forward()
    rects = []
    
    for i in np.arange(detections.shape[2]):
        conf = detections[0, 0, i, 2]

        if conf > CONF:
            id = int(detections[0, 0, i, 1])
            if CLASSES[id] in FOCUS_CLASSES:
                (startX, startY, endX, endY) = (detections[0, 0, i, 3:7] * np.array([W, H, W, H]) * np.array([rW, rH, rW, rH])).astype('int')
                rects.append((startX, startY, endX, endY))
                cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[id], 2)

    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv2.putText(orig, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(orig, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    cv2.imshow('show', orig)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()