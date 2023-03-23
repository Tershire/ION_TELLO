"""
ArUcoTools.py
class for ArUco Markers

1st written by: Wonhee Lee
1st written on: 2023 MAR 09
guided by: https://python-academia.com/en/opencv-aruco/
           https://stackoverflow.com/questions/74964527/attributeerror-module-cv2-aruco-has-no-attribute-dictionary-get
           https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python/blob/main/pose_estimation.py
           https://gist.github.com/edward1986/ed1ae7f8a10a3d44dc82360a38c4c6b9
"""

# IMPORT //////////////////////////////////////////////////////////////////////
import cv2 as cv
from cv2 import aruco
import numpy as np
import time
import threading
import keyboard


# SETTING /////////////////////////////////////////////////////////////////////


# CLASS ///////////////////////////////////////////////////////////////////////
class DetectMarker:
    """
    continuously detect ArUco Marker viewed on camera
    """
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()

    def __init__(self, camera_id):
        self.cap = cv.VideoCapture(camera_id)

    def get_marker_info(self):
        """
        get marker id list
        :return:
        """
        ret, frame = self.cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected_img_points = detector.detectMarkers(gray)
        ids = np.ravel(ids)

        return corners, ids, self.cap

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
            # get ArUco info
            corners, ids, _ = detect_marker_cam0.get_marker_info()

            ret, frame = cap.read()
            aruco.drawDetectedMarkers(frame, corners)
            print("get_marker_id_cam0")
            print(ids)
            # aruco.drawAxis(frame, )
            cv.imshow('View', frame)
            cv.waitKey(1)

            if self.stopped():
                cap.release()
                cv.destroyAllWindows()
                break


# TEST ////////////////////////////////////////////////////////////////////////
if __name__ == "__main__":
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()

    # create object
    camera_id = 0
    detect_marker_cam0 = DetectMarker(camera_id)

    _, _, cap = detect_marker_cam0.get_marker_info()    # what is better?
    streamer = ArUcoStreamer(cap)

    # start Thread
    streamer.start()

    """
    except KeyboardInterrupt:
        detect_marker_cam0.cap.release()
    """

    # input("Press Any Key to Stop")  # -> this not good method

    while (not keyboard.read_key()):
        print("running")

    # finish Thread
    streamer.stop()
    streamer.join()
