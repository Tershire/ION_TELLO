"""
ArUcoTools.py
class for ArUco Markers

1st written by: Wonhee Lee
1st written on: 2023 MAR 24 based on legacy code now named ArUcoTools_old2.py
guided by: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
           https://python-academia.com/en/opencv-aruco/
           https://stackoverflow.com/questions/74964527/attributeerror-module-cv2-aruco-has-no-attribute-dictionary-get
           https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python/blob/main/pose_estimation.py
           https://gist.github.com/edward1986/ed1ae7f8a10a3d44dc82360a38c4c6b9
"""

# IMPORT //////////////////////////////////////////////////////////////////////
import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import threading
import keyboard


# SETTING /////////////////////////////////////////////////////////////////////
# ArUco -----------------------------------------------------------------------
# dictionary choice
aruco_dict = aruco.DICT_4X4_50
markerLength = 0.175

# camera ----------------------------------------------------------------------
camera_id = 0

# temporary trials: actual calibration process must be implementedz
cameraMatrix = np.array([[543.05833681, 0., 326.0951866],
                         [0., 542.67378833, 247.65515938],
                         [0., 0., 1.]])
distCoeffs = np.array([-0.28608759, 0.13647301, -0.00076189, 0.0014116, -0.06865808])


# CLASS ///////////////////////////////////////////////////////////////////////
class MarkerTracker:
    """
    detect and estimate pose of each ArUco Marker
    """
    def __init__(self, cap, aruco_dict, markerLength, cameraMatrix, distCoeffs):
        self.cap = cap
        self.aruco_dict = aruco_dict
        self.markerLength = markerLength
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

        dictionary = aruco.getPredefinedDictionary(aruco_dict)
        parameters = aruco.DetectorParameters()
        self.dictionary = dictionary
        self.parameters = parameters

        # marker corners
        objectPoints0 = np.array([-markerLength / 2, +markerLength / 2, 0])
        objectPoints1 = np.array([+markerLength / 2, +markerLength / 2, 0])
        objectPoints2 = np.array([+markerLength / 2, -markerLength / 2, 0])
        objectPoints3 = np.array([-markerLength / 2, -markerLength / 2, 0])
        objectPoints = np.array([objectPoints0, objectPoints1,
                                 objectPoints2, objectPoints3])
        self.objectPoints = objectPoints

    def get_marker_info(self):
        """
        get marker id list
        """
        ret, frame = self.cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        # detect marker
        detector = aruco.ArucoDetector(self.dictionary, self.parameters)
        corners, ids, rejected_img_points = detector.detectMarkers(gray)
        ids = np.ravel(ids)

        # draw detected marker
        # aruco.drawDetectedMarkers(frame, corners)  # aruco.drawDetectedMarkers(frame, corners, ids): NOT working

        return corners, ids

    def get_pose(self, corners, ids):
        """
        calculate each marker pose and draw corresponding axes
        """
        ret, frame = self.cap.read()

        rvecs, tvecs = {}, {}
        if ids[0] is not None:
            for i, id in enumerate(ids):
                # calculate pose for each marker
                ret, rvec, tvec = cv.solvePnP(self.objectPoints, corners[i],
                                              self.cameraMatrix, self.distCoeffs)

                rvecs[id] = rvec
                tvecs[id] = tvec

                # draw axis for each marker
                cv.drawFrameAxes(frame, self.cameraMatrix, self.distCoeffs,
                                 rvec, tvec, 0.1)

        return rvecs, tvecs, frame


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
            # get marker info
            corners, ids = marker_tracker.get_marker_info()
            print("ids:", ids)
            # aruco.drawAxis(frame, )

            # get pose
            _, _, frame = marker_tracker.get_pose(corners, ids)

            # draw
            aruco.drawDetectedMarkers(frame, corners)

            cv.imshow('View', frame)
            cv.waitKey(1)

            if self.stopped():
                cap.release()
                cv.destroyAllWindows()
                break


# TEST ////////////////////////////////////////////////////////////////////////
if __name__ == "__main__":
    # create object
    cap = cv.VideoCapture(camera_id)

    marker_tracker = MarkerTracker(cap, aruco_dict, markerLength,
                                   cameraMatrix, distCoeffs)

    # create Thread
    streamer = ArUcoStreamer(cap)

    # start Thread
    streamer.start()

    # wait for keyboard
    keyboard.wait("esc")
    # while not keyboard.read_key():
    #     pass

    # finish Thread
    streamer.stop()
    streamer.join()
