import cv2
import numpy as np
import math
import opticScan

debug = True

def show(image, name):
    if debug:
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
source = cv2.imread("deneme8.jpg")
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

# Detecting corners of the sheet.
perimeter = cv2.arcLength(cnt, True)
epsilon = 0.02 * perimeter
approxCorners = cv2.approxPolyDP(cnt, epsilon, True)
approxCornersNumber = len(approxCorners)
array = np.array([approxCorners[0][0], approxCorners[1][0], approxCorners[2][0], approxCorners[3][0]])
sorted_pts = sort_pts(array)

marked = img.copy()
for point in sorted_pts:
    cv2.drawMarker(marked, (int(point[0]), int(point[1])), (255, 0, 255), cv2.MARKER_STAR)
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

# Additional orientation fix.
codeStrap = colorMasked[0:842, 0:45].copy()
show(codeStrap, "cropped")

canny_codeStrap = cv2.Canny(codeStrap, 10, 50)
show(canny_codeStrap, "canny2")

contours_codeStrap = cv2.findContours(canny_codeStrap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
contoured_codeStrap = cv2.drawContours(codeStrap.copy(), contours_codeStrap, -1, (0, 255, 0), 1)
show(contoured_codeStrap, "contoured_codeStrap")

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
xCenter = sum(pointsX)/len(pointsX)
yCenter = sum(pointsY)/len(pointsY)
image_center = (xCenter, yCenter)
rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
fixed = cv2.warpAffine(colorMasked.copy(), rot_mat, colorMasked.shape[1::-1], flags=cv2.INTER_CUBIC)
show(fixed, "fixed")

######################################

# Calculating grid spaces and starting coordinates based on code strap.
codeStrap = colorMasked[0:842, 0:45].copy()

canny_codeStrap = cv2.Canny(codeStrap, 10, 50)

contours_codeStrap = cv2.findContours(canny_codeStrap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
contoured_codeStrap = cv2.drawContours(codeStrap.copy(), contours_codeStrap, -1, (0, 255, 0), 1)

scanned = fixed.copy()
pointsX = []
pointsY = []

# Finding positions of rectangles in code strap.
for contour in contours_codeStrap:
    moment = cv2.moments(contour)
    cX = (moment["m10"] / moment["m00"])
    cY = (moment["m01"] / moment["m00"])
    pointsX.append(cX)
    pointsY.append(cY)

# Calculating average space between rectangles in code strap.
ySum = 0
for i in range(len(pointsY) - 1):
    ySum += (pointsY[i] - pointsY[i + 1])

xSpace = float(ySum / (len(pointsY) - 1))

# Calculating horizontal starting point and column count for the scan.
xSum = 0
for i in pointsX:
    xSum += i

xStart = xSum / len(pointsX)
xNum = int((595 - xStart) / xSpace)


# Scanning the grid.
grid = []
for yIndex in range(len(pointsY)):
    gridX = []
    for xIndex in range(1, xNum):
        average = opticScan.checkSquare(fixed, int(xStart + xIndex * xSpace - xSpace / 2),
                                        int(pointsY[yIndex] - xSpace / 2), int(xSpace))
        if average > 75:
            color = (255, 0, 255)
            cv2.drawMarker(scanned, (int(xStart + xIndex * xSpace), int(pointsY[yIndex])), color, cv2.MARKER_SQUARE,
                           thickness=1, markerSize=int(xSpace))
            gridX.append(1)
        else:
            color = (0, 255, 0)
            gridX.append(0)
    grid.insert(0, gridX)

show(scanned, "scanned")

print(grid)
