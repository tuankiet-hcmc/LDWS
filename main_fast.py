# -*- coding: utf-8 -*-
from imutils.video import FPS
import time
from detect_lane import *
from file_video_stream_fast import FileVideoStream
from preprocessing_image import *
# from blink import *

print("[INFO] starting video file thread...")
# cap1 = FileVideoStream("test_videos/test1.mp4").start()
# cap2 = FileVideoStream("test_videos/test2.mp4").start()
cap3 = FileVideoStream("test_videos/test3.mp4").start()
time.sleep(0.1)
fps = FPS().start()


def process_video(video_input, video_output):
    detector = LaneDetector()

    clip = VideoFileClip(os.path.join('test_videos', video_input))
    processed = clip.fl_image(detector.process)
    processed.write_videofile(os.path.join(
        'output_videos', video_output), audio=False)


if __name__ == '__main__':
    # setup()
    while cap3.more():
        # frame1 = cap1.read()
        # frame2 = cap2.read()
        (frame3, preprocessed_image) = cap3.read()

        # frame1 = imutils.resize(frame1, width=450)
        # frame2 = imutils.resize(frame2, width=450)
        # frame3 = imutils.resize(frame3, width=450)

        # detector1 = LaneDetector()
        # processed_img1 = detector1.process(frame1, False)

        # detector2 = LaneDetector()
        # processed_img2 = detector2.process(frame2, False)

        preprocessed_image = preprocess_image(preprocessed_image, True)

        detector3 = LaneDetector()
        processed_img3 = detector3.process(frame3, preprocessed_image)

        # processed_img1 = preprocess_image(frame1)
        # processed_img2 = preprocess_image(frame2)

        # lines1 = hough_lines(processed_img1)
        # processed_img1 = draw_lines(frame1, lines1)

        # lines2 = hough_lines(processed_img2)
        # if lines2 is not None:
        #     processed_img2 = draw_lines(frame2, lines2)
        

        # cv2.imshow("Video 1", processed_img1)
        # cv2.imshow("Video 2", processed_img2)
        cv2.imshow("Video 3", processed_img3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps.update()
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cap1.stop()
    # cap2.stop()
    cap3.stop()
    cv2.destroyAllWindows()
    # destroy()
