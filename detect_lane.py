from collections import deque
from hough_transform import *
# from blink import *
QUEUE_LENGTH = 128


class LaneDetector:
    def __init__(self):
        self.left_lines = deque(maxlen=QUEUE_LENGTH)
        self.right_lines = deque(maxlen=QUEUE_LENGTH)

    def process(self, origin_image, preprocessed_image):
        lines = hough_lines(preprocessed_image)
        left_line, right_line = lane_lines(origin_image, lines)

        font = cv2.FONT_HERSHEY_SIMPLEX

        def mean_line(line, lines):
            if line is not None:
                lines.append(line)

            if len(lines) > 0:
                line = np.mean(lines, axis=0, dtype=np.int32)
                # make sure it's tuples not numpy array for cv2.line to work
                line = tuple(map(tuple, line))
            return line

        left_line = mean_line(left_line,  self.left_lines)
        right_line = mean_line(right_line, self.right_lines)
        if left_line is not None and right_line is not None:
            (xl1, yl1), (xl2, yl2) = left_line
            (xr1, yr1), (xr2, yr2) = right_line
            if xr2 - xl2 < 50 : 
                return draw_lane_lines(origin_image, (None, None), color=[0, 0, 0], thickness=0)
            al = yl1 - yl2
            bl = xl2 - xl1
            ar = yr1 - yr2
            br = xr1 - xr2
            dlx, dly = xl2 - xl1, yl1 - yl2
            drx, dry = xr1 - xr2, yr1 - yr2
            anglel = np.arctan2(dly, dlx) * (180/np.pi)
            angler = np.arctan2(dry, drx) * (180/np.pi)
            if anglel > 60:
                cv2.putText(origin_image, "Left", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # red()
            elif angler > 60:
                cv2.putText(origin_image, "Right", (origin_image.shape[1] - 60, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # white()
        elif left_line is None and right_line is not None:
            (xr1, yr1), (xr2, yr2) = right_line
            ar = yr1 - yr2
            br = xr1 - xr2
            drx, dry = xr1 - xr2, yr1 - yr2
            angler = np.arctan2(dry, drx) * (180/np.pi)
            if angler > 60:
                cv2.putText(origin_image, "Right", (origin_image.shape[1] - 60, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # white()
        elif right_line is None and left_line is not None:
            (xl1, yl1), (xl2, yl2) = left_line
            al = yl1 - yl2
            bl = xl2 - xl1
            dlx, dly = xl2 - xl1, yl1 - yl2
            anglel = np.arctan2(dly, dlx) * (180/np.pi)
            if xl2 > origin_image.shape[1]/2:
                return draw_lane_lines(origin_image, (None, None), color=[0, 0, 0], thickness=0)
            if anglel > 60:
                cv2.putText(origin_image, "Left", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # red()

        return draw_lane_lines(origin_image, (left_line, right_line), thickness=10)
