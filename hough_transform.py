from __future__ import division
import cv2
import numpy as np


def hough_lines(image):
    """
    `image` should be the output of a Canny transform.

    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=10, minLineLength=20,maxLineGap=500)


def draw_lines(image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
    # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
    if make_copy:
        image = np.copy(image)  # don't want to modify the original
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


def average_slope_intercept(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue  # ignore a vertical line
                if y2 == y1:
                    continue  # ignore a vertical line
                slope = (y2-y1)/(x2-x1)
                intercept = y1 - slope*x1
                length = np.sqrt((y2-y1)**2+(x2-x1)**2)
                anglel = np.arctan2(abs((y2-y1)), abs((x2-x1))) * (180/np.pi)

                if anglel < 20:
                    continue

                if slope < 0:  # y is reversed in image
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))
    except:
        pass

    # add more weight to longer lines
    left_lane = np.dot(left_weights,  left_lines) / \
        np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / \
        np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane  # (slope, intercept), (slope, intercept)


def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line

    # make sure everything is integer as cv2.line requires it

    x1 = (y1 - intercept)//slope
    x2 = (y2 - intercept)//slope
    y1 = (y1)
    y2 = (y2)

    if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
        return None

    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    imshape = image.shape
    A = (0.42*imshape[1], 0.53*imshape[0])
    B = (0.58*imshape[1], 0.53*imshape[0])
    C = (0.82*imshape[1], 0.75*imshape[0])
    D = (0.25*imshape[1], 0.75*imshape[0])

    left_lane, right_lane = average_slope_intercept(lines)

    y1 = image.shape[0]*0.78  # bottom of the image
    y2 = image.shape[0]*0.55         # slightly lower than the middle

    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    if left_line is not None and right_line is not None:
        (xl1, yl1), (xl2, yl2) = left_line
        (xr1, yr1), (xr2, yr2) = right_line
        xa, ya = A
        xb, yb = B
        if xr2 - xl2 < (xb-xa - 150*(imshape[1]/1280)):
            left_line = None
            right_lane = None

    if left_line is not None:
        (xl1, yl1), (xl2, yl2) = left_line
        if xl2 > 640*(imshape[1]/1280):
            left_line = None
    if right_line is not None:
        (xr1, yr1), (xr2, yr2) = right_line
        if xr2 < 600*(imshape[1]/1280):
            right_line = None
        if xr1 < xr2:
            right_line = None

    return left_line, right_line


def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color=color, thickness=thickness)
    # image1 * a + image2 * b + lamda
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)
