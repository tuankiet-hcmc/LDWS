# -*- coding: utf-8 -*-
from imutils.video import FileVideoStream
from imutils.video import FPS
import imutils
import time
from detect_lane import *
from preprocessing_image import *
from lane_marking import *

cap = cv2.VideoCapture("test_videos/test3.mp4")
fps = FPS().start()

if __name__ == '__main__':
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=450)
        preprocessed_image = preprocess_image(frame)
        lane_marking_image = lane_marking_detection(preprocessed_image, True)

        detector = LaneDetector()
        processed_img = detector.process(frame, lane_marking_image)

        cv2.imshow("Screen", processed_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps.update()
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cap.release()
    cv2.destroyAllWindows()
