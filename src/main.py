import cv2
import numpy as np
import math
import opticScan


def show(image, name):
    cv2.imshow(name, image)
    cv2.waitKey(-1)
    cv2.destroyWindow(name)


def sort_pts(points):
    sorted_points = np.zeros((4, 2), dtype=np.float32)
    s = np.sum(points, axis=1)
    sorted_points[0] = points[np.argmin(s)]
    sorted_points[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    sorted_points[1] = points[np.argmin(diff)]
    sorted_points[3] = points[np.argmax(diff)]

    return sorted_points


# Reading image and downscaling.
source = cv2.imread("deneme7.jpg")
ratio = float(source.shape[1] / 270)
img = cv2.resize(source, (270, 480), interpolation=cv2.INTER_AREA)

# Applying grayscale, blur and canny filters.
grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show(grayScale, "gray")

blurred = cv2.medianBlur(grayScale, 7)
show(blurred, "blur")

canny = cv2.Canny(blurred, 50, 255)
show(canny, "canny")

# Detecting edges and the contour of the sheet.
ret, thresh = cv2.threshold(canny, 25, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contoured = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 1)
show(contoured, "contour")

areas = []
for i in range(len(contours)):
    if i == 0:
        pass
    areas.append(cv2.contourArea(contours[i]))
max_index = np.argmax(areas)
cnt = contours[max_index]
"""x, y, w, h = cv2.boundingRect(cnt)
bounding = cv2.rectangle(img.copy(), (x, y), (x+w, y+h), (0, 255, 0), 2)
show(bounding, "bounding")"""

# Detecting corners of the sheet.
perimeter = cv2.arcLength(cnt, True)
epsilon = 0.02 * perimeter
approxCorners = cv2.approxPolyDP(cnt, epsilon, True)
approxCornersNumber = len(approxCorners)
array = np.array([approxCorners[0][0], approxCorners[1][0], approxCorners[2][0], approxCorners[3][0]])
sorted_pts = sort_pts(array)

marked = img.copy()
for point in sorted_pts:
    cv2.drawMarker(marked, (point[0], point[1]), (255, 0, 255), cv2.MARKER_STAR)
show(marked, "marked")

# Warping sheet to full size.
pts1 = sorted_pts * ratio
pts2 = np.float32([[0, 0], [595, 0], [595, 842], [0, 842]])
transformMatrix = cv2.getPerspectiveTransform(pts1, pts2)
warped = cv2.warpPerspective(source, transformMatrix, (595, 842), flags=cv2.INTER_CUBIC)
show(warped, "warped")

# Applying color mask to sheet.
colorMasked = opticScan.editColor(warped).copy()
show(colorMasked, "color")

codeStrap = colorMasked[0:842, 0:45].copy()
show(codeStrap, "cropped")

canny_codeStrap = cv2.Canny(codeStrap, 10, 50)
show(canny_codeStrap, "canny2")

contours_codeStrap = cv2.findContours(canny_codeStrap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
contoured_codeStrap = cv2.drawContours(codeStrap.copy(), contours_codeStrap, -1, (0, 255, 0), 1)
show(contoured_codeStrap, "contour2")

pointsX = []
pointsY = []
for contour in contours_codeStrap:
    moment = cv2.moments(contour)
    cX = (moment["m10"] / moment["m00"])
    cY = (moment["m01"] / moment["m00"])
    pointsX.append(cX)
    pointsY.append(cY)
"""print(len(contours2))"""
xArr = np.array(pointsX)
yArr = np.array(pointsY)
m, b = np.polyfit(xArr, yArr, 1)
angle = ((math.atan(m) / math.pi) * 180)
if angle >= 0:
    angle = angle - 90
else:
    angle = 90 + angle
pointsY, pointsX = zip(*sorted(zip(pointsY, pointsX)))
image_center = (pointsX[32], pointsY[32])
rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
result = cv2.warpAffine(colorMasked.copy(), rot_mat, colorMasked.shape[1::-1], flags=cv2.INTER_CUBIC)
show(result, "result")

######################################

cropped2 = colorMasked[0:842, 0:45].copy()

canny3 = cv2.Canny(cropped2, 10, 50)

contours3 = cv2.findContours(canny3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
contoured3 = cv2.drawContours(cropped2.copy(), contours3, -1, (0, 255, 0), 1)

noktalarX = []
noktalarY = []
lined = result.copy()
for contour in contours3:
    moment = cv2.moments(contour)
    cX = (moment["m10"] / moment["m00"])
    cY = (moment["m01"] / moment["m00"])
    noktalarX.append(cX)
    noktalarY.append(cY)
    cY = int(cY)
    """cv2.line(lined,(0,cY),(595,cY),(0,255,0))"""

xSpace = noktalarY[9] - noktalarY[10]
print(xSpace)
ySum = 0
for i in range(len(noktalarY) - 1):
    ySum += (noktalarY[i] - noktalarY[i + 1])
xSpace = float(ySum / (len(noktalarY) - 1))

xSum = 0
for i in noktalarX:
    xSum += i
xStart = xSum / len(noktalarX)

xNum = int((595 - xStart) / xSpace)
print(f"space: {xSpace}")
"""for i in range(xNum):
    cv2.line(lined, (int(i*xSpace+xStart), 0), (int(i*xSpace+xStart), 842), (0, 255, 0))"""

grid = []
for yIndex in range(len(noktalarY)):
    gridX = []
    for xIndex in range(1, xNum):
        average = opticScan.checkSquare(result, int(xStart + (xIndex) * xSpace - xSpace / 2),
                                        int(noktalarY[yIndex] - xSpace / 2), int(xSpace))
        print(average)
        if average > 75:
            color = (255, 0, 255)
            cv2.drawMarker(lined, (int(xStart + (xIndex) * xSpace), int(noktalarY[yIndex])), color, cv2.MARKER_SQUARE,
                           thickness=1, markerSize=int(xSpace))
            gridX.append(1)
        else:
            color = (0, 255, 0)
            gridX.append(0)
    grid.insert(0, gridX)

show(lined, "SONN")

print(grid)
