"""
ArUcoTools.py
class for ArUco Markers

1st written by: Wonhee Lee
1st written on: 2023 MAR 09
guided by: https://python-academia.com/en/opencv-aruco/
           https://stackoverflow.com/questions/74964527/attributeerror-module-cv2-aruco-has-no-attribute-dictionary-get

"""

# IMPORT //////////////////////////////////////////////////////////////////////
import cv2 as cv
from cv2 import aruco
import numpy as np
import time
import vision_tools as vt


# SETTING /////////////////////////////////////////////////////////////////////


# CLASS ///////////////////////////////////////////////////////////////////////
class DetectMarker:
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()

    def __init__(self, camera_id):
        self.cap = cv.VideoCapture(camera_id)

    def get_marker_id(self):
        """
        get marker id list from an image
        :return:
        """
        ret, frame = self.cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected_img_points = detector.detectMarkers(gray)
        ids = np.ravel(ids)

        return ids, self.cap


# Test
if __name__ == "__main__":
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()

    # create object
    camera_id = 0
    detect_marker_cam0 = DetectMarker(camera_id)

    _, cap = detect_marker_cam0.get_marker_id()
    streamer = vt.CapStreamer(cap)

    # start Thread
    streamer.start()

    try:
        while True:
            print("get_marker_id_cam0")
            ids, _ = detect_marker_cam0.get_marker_id()
            print(ids)
            time.sleep(1)

    except KeyboardInterrupt:
        detect_marker_cam0.cap.release()

    # finish Thread
    streamer.stop()
    streamer.join()
