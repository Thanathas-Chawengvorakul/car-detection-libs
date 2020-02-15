import LicensePlate as lp 
import cv2

"""
LicensePlate Lib 
- markLicense
  @Params@
    img: image(color), 
    color=(x,y,z) bgr color, 
    thickness
  @Return@
    img that is already marked

- posLicense
  @Params@
    img: image(color)
  @Reuturn@ 
    list of rectangle that contain license plate
"""
img = cv2.imread('datasets/lp.jpg')
lpImg = lp.markLicense(img)
lpList = lp.posLicense(img)
print(lpList)
cv2.imshow('s', lpImg)
cv2.waitKey(0)