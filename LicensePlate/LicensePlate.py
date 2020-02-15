from cv2 import cvtColor, arcLength, approxPolyDP, GaussianBlur, adaptiveThreshold, threshold, findContours, rectangle, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, THRESH_BINARY, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE
from numpy import array
from imutils import grab_contours

def detect(c):
    peri = arcLength(c, 1)
    approx = approxPolyDP(c, 0.04 * peri, 1)

    return len(approx) == 4

def markLicense(img, color=(255, 0, 0), thickness=2):
    gray = cvtColor(img, 6)
    imgBlurred = GaussianBlur(gray, (5, 5), 0)
    imgThresh = adaptiveThreshold(imgBlurred, 255.0, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 19, 9)
    thresh = threshold(imgBlurred, 60, 255, THRESH_BINARY)[1]
    cnts = findContours(thresh.copy(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)

    for c in cnts:
        if detect(c):
            reshaped = c.reshape(-1, 2)
            leftUpXy = tuple(map(min, zip(*reshaped)))
            rightDownXy = tuple(map(max, zip(*reshaped)))
            rectangle(img, leftUpXy, rightDownXy, color, thickness)
    return img

def posLicense(img):
    gray = cvtColor(img, 6)
    imgBlurred = GaussianBlur(gray, (5, 5), 0)
    imgThresh = adaptiveThreshold(imgBlurred, 255.0, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 19, 9)
    thresh = threshold(imgBlurred, 60, 255, THRESH_BINARY)[1]
    cnts = findContours(thresh.copy(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)

    posList = []
    for c in cnts:

        if detect(c):
            reshaped = c.reshape(-1, 2)
            posList.append((tuple(map(min, zip(*reshaped))), tuple(map(max, zip(*reshaped)))))
    
    return array(posList)