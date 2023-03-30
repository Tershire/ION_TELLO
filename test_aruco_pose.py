"""
test_aruco_pose.py
get real world ArUco Marker pose

1st written by: Wonhee Lee
1st written on: 2023 MAR 30
referred: https://stackoverflow.com/questions/34588464/python-how-to-capture-image-from-webcam-on-click-using-opencv
"""

# IMPORT //////////////////////////////////////////////////////////////////////
import cv2 as cv
import numpy as np
import threading
import keyboard
import ArUcoTools as a_t


# SETTING /////////////////////////////////////////////////////////////////////
# ArUco -----------------------------------------------------------------------
# dictionary choice
aruco_dict = cv.aruco.DICT_4X4_50
markerLength = 0.175

# camera ----------------------------------------------------------------------
camera_id = 0

# temporary trials: actual calibration process must be implementedz
cameraMatrix = np.array([[543.05833681, 0., 326.0951866],
                         [0., 542.67378833, 247.65515938],
                         [0., 0., 1.]])
distCoeffs = np.array([-0.28608759, 0.13647301, -0.00076189, 0.0014116, -0.06865808])


# CLASS ///////////////////////////////////////////////////////////////////////
class ArUcoStreamer(threading.Thread):
    """
    "Killable Thread" to show live stream
    """
    def __init__(self, cap):
        super(ArUcoStreamer, self).__init__()
        self._stop_event = threading.Event()
        self.cap = cap

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        self.stream_video(self.cap)

    def stream_video(self, cap):
        # stream a video
        while True:
            ret, frame = cap.read()
            cv.imshow('View', frame)
            k = cv.waitKey(1)

            if k % 256 == 27:    # esc
                break

            elif k % 256 == 32:    # space
                # get marker info
                corners, ids = marker_tracker.get_marker_info()
                print("ids:", ids)
                # aruco.drawAxis(frame, )

                # get pose
                frame = marker_tracker.get_pose(corners, ids)

                # draw
                cv.aruco.drawDetectedMarkers(frame, corners)

                cv.imshow('View', frame)
                cv.waitKey(1)

            if self.stopped():
                cap.release()
                cv.destroyAllWindows()
                break


# EVENT LOOP //////////////////////////////////////////////////////////////////
# create object
cap = cv.VideoCapture(camera_id)

marker_tracker = a_t.MarkerTracker(cap, aruco_dict, markerLength,
                               cameraMatrix, distCoeffs)

# create Thread
streamer = ArUcoStreamer(cap)

# start Thread
streamer.start()

# wait for keyboard
keyboard.wait("esc")

# finish Thread
streamer.stop()
streamer.join()
