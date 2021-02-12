import cv2
import numpy as np


def checkSquare(img,x,y,size):
    crop_img = img[y:y+size, x:x + size]
    avg = cv2.mean(crop_img)
    return avg[0]+avg[1]+avg[2]


def editColor(var):
    img = var
    blank = np.zeros((var.shape[0], var.shape[1], 3), np.uint8)
    blank[:] = (255, 255, 255)
    #img = cv2.resize(img, (var.shape[1], var.shape[0]), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #img = cv2.addWeighted(img, 1.25, img, 0, 1)

    """cv2.imshow("img",img)
    cv2.waitKey(-1)"""

    imgHSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    """cv2.imshow("HSV",imgHSV)
    cv2.waitKey(-1)"""

    lowerPink = np.array([145, 30, 20])
    upperPink = np.array([168, 255, 255])

    black1 = np.array([0,0,0])
    black2 = np.array([180,255,100])

    mask = cv2.inRange(imgHSV, black1, black2)
    """mask = cv2.inRange(imgHSV, lowerPink, upperPink, mask)"""
    result = cv2.bitwise_xor(blank, img, mask=mask)
    """cv2.imshow("sonuc",result)
    cv2.waitKey(-1)"""

    blur = cv2.blur(result, (1,1))

    """cv2.imshow("blur",blur)
    cv2.waitKey(-1)"""
    return blur
