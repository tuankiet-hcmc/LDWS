import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

import numpy as np

def preprocess_image(image, color=True):

    # image = adjust_gamma(image, gamma=1.5)

    image = denoise(image)
    if color is True:
        image = color_selection(image)

    image = cv2.Canny(image, 30, 60, apertureSize = 3)

    image = region_of_interest(image)

    return image

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def denoise(frame):
    frame = cv2.medianBlur(frame, 5)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    return frame

def region_of_interest(img):
    imshape = img.shape
    A = (0.42*imshape[1], 0.53*imshape[0])
    B = (0.58*imshape[1], 0.53*imshape[0])
    C = (0.82*imshape[1], 0.75*imshape[0])
    D = (0.25*imshape[1], 0.75*imshape[0])
    vertices = np.array([[B, C, D, A]])
    # defining a blank mask to start with
    mask = np.zeros_like(img, dtype=np.uint8)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), ignore_mask_color)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

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
