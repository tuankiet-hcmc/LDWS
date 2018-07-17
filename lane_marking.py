import cv2
import numpy as np

def lane_marking_detection(image, color=True):
    if color is True:
        image = color_selection(image)
    image = cv2.Canny(image, 30, 60, apertureSize = 3)
    return image

def color_selection(image):
    imaged1 = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    imaged2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.uint8([5, 150, 15])
    upper = np.uint8([20, 200, 30])

    mark1 = cv2.inRange(imaged1, lower, upper)

    lower = np.uint8([5, 50, 100])
    upper = np.uint8([30, 255, 200])

    mark2 = cv2.inRange(imaged2, lower, upper)

    mask = cv2.bitwise_or(mark1, mark2)

    return cv2.bitwise_and(image, image, mask = mask)