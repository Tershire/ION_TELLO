"""
ArUcoTools.py
class for ArUco Markers

1st written by: Wonhee Lee
1st written on: 2023 MAR 09
guided by: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
           https://python-academia.com/en/opencv-aruco/
           https://stackoverflow.com/questions/74964527/attributeerror-module-cv2-aruco-has-no-attribute-dictionary-get
           https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python/blob/main/pose_estimation.py
           https://gist.github.com/edward1986/ed1ae7f8a10a3d44dc82360a38c4c6b9
"""

# IMPORT //////////////////////////////////////////////////////////////////////
import cv2 as cv
from cv2 import aruco
import numpy as np
import threading
import keyboard


# SETTING /////////////////////////////////////////////////////////////////////
# ArUco dictionary choice
aruco_dict = aruco.DICT_4X4_50


# CLASS ///////////////////////////////////////////////////////////////////////
class MarkerDetector:
    """
    detect ArUco Marker viewed on camera
    """
    dictionary = aruco.getPredefinedDictionary(aruco_dict)
    parameters = aruco.DetectorParameters()

    def __init__(self, cap):
        self.cap = cap

    def get_marker_info(self, cap):
        """
        get marker id list
        :return:
        """
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected_img_points = detector.detectMarkers(gray)
        ids = np.ravel(ids)

        return corners, ids


class PoseEstimator:
    """
    estimate pose of a marker
    """
    dictionary = aruco.getPredefinedDictionary(aruco_dict)
    parameters = aruco.DetectorParameters()

    def __init__(self, cap, markerLength, cameraMatrix, distCoeffs):
        self.cap = cap
        self.markerLength = markerLength
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

        # marker corners
        objectPoints0 = np.array([-self.markerLength / 2,  self.markerLength / 2, 0])
        objectPoints1 = np.array([ self.markerLength / 2,  self.markerLength / 2, 0])
        objectPoints2 = np.array([ self.markerLength / 2, -self.markerLength / 2, 0])
        objectPoints3 = np.array([-self.markerLength / 2, -self.markerLength / 2, 0])
        self.objectPoints = np.array([objectPoints0, objectPoints1, objectPoints2, objectPoints3])

    def get_pose(self, cap, corners, ids, cameraMatrix, distCoeffs):
        """
        calculate each marker pose and draw corresponding axes
        """
        ret, frame = cap.read()

        if ids[0] is not None:
            for i in range(len(ids)):
                # calculate pose for each marker
                ret, rvec, tvec = cv.solvePnP(self.objectPoints, corners[i], cameraMatrix, distCoeffs)

                print("rvec, tvec:", rvec, tvec)

                # draw axis for each marker
                cv.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.1)

        return frame


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

            # get ArUco info
            corners, ids = marker_detector.get_marker_info(cap)
            print("ids:", ids)
            # aruco.drawAxis(frame, )

            frame = pose_estimator.get_pose(cap, corners, ids, cameraMatrix, distCoeffs)
            aruco.drawDetectedMarkers(frame, corners)    # aruco.drawDetectedMarkers(frame, corners, ids): NOT working

            cv.imshow('View', frame)
            cv.waitKey(1)

            if self.stopped():
                cap.release()
                cv.destroyAllWindows()
                break


# TEST ////////////////////////////////////////////////////////////////////////
if __name__ == "__main__":
    dictionary = aruco.getPredefinedDictionary(aruco_dict)
    parameters = aruco.DetectorParameters()

    # create object
    camera_id = 0
    cap = cv.VideoCapture(camera_id)

    # temporary trials: actual calibration process must be implemented
    markerLength = 0.175
    cameraMatrix = np.array([[543.05833681, 0.,           326.0951866 ],
                             [0.,           542.67378833, 247.65515938],
                             [0.,           0.,           1.          ]])
    distCoeffs = np.array([-0.28608759, 0.13647301, -0.00076189, 0.0014116, -0.06865808])

    pose_estimator = PoseEstimator(cap, markerLength, cameraMatrix, distCoeffs)
    marker_detector = MarkerDetector(cap)

    # create Thread
    streamer = ArUcoStreamer(cap)

    # start Thread
    streamer.start()

    # wait for keyboard
    while not keyboard.read_key():
        pass

    # finish Thread
    streamer.stop()
    streamer.join()
