import LicensePlate as lp 
import cv2

img = cv2.imread('datasets/lp.jpg')
lpImg = lp.markLicense(img)
lpList = lp.posLicense(img)
print(lpList)
cv2.imshow('s', lpImg)
cv2.waitKey(0)