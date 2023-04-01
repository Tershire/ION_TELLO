"""
follow_aruco.py
Tello Edu drone follows an ArUco marker

1st written by: Wonhee Lee
1st written on: 2023 APR 01
"""

# IMPORT //////////////////////////////////////////////////////////////////////
import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import threading
import keyboard
import ArUcoTools as aT
import matplotlib.pyplot as plt
import CalibrationTools as cT
from djitellopy import Tello

# SETTING /////////////////////////////////////////////////////////////////////
# ArUco -----------------------------------------------------------------------
# dictionary choice
aruco_dict = aruco.DICT_4X4_50
markerLength = 17.5E-2  # [cm] marker side length

# camera ----------------------------------------------------------------------
camera_name = 'tello'

# intrinsics
calib_file_name = r'C:\Users\leewh\Documents\Academics\Research\FR\Drone\Calibration_Data\\' + \
                  f'{camera_name}_intrinsics.yml'
cameraMatrix, distCoeffs = cT.load_camera_intrinsics(calib_file_name)

# plot ------------------------------------------------------------------------
fig = plt.figure(figsize=plt.figaspect(1 / 3))
axs = []
for i in range(3):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    axs.append(ax)

# plot camera pose
posi = np.array([0, 0, 0])
dire = np.array([0, 0, 1])

# quiver
for ax in axs:
    ax.quiver(posi[0], posi[1], posi[2], dire[0], dire[1], dire[2], color='k')

# setting
for ax in axs:
    ax.set_xlim([-0.5, +0.5])
    ax.set_ylim([-0.5, +0.5])
    ax.set_zlim([0, +2])

    ax.set_xlabel('x_{c}')
    ax.set_ylabel('y_{c}')
    ax.set_zlabel('z_{c}')

axs[0].view_init(elev=0, azim=-90, roll=180)
axs[1].view_init(elev=0, azim=0, roll=-90)
axs[2].view_init(elev=+90, azim=+90)


# CLASS ///////////////////////////////////////////////////////////////////////
class VisionThread(threading.Thread):
    """
    "Killable Thread" to show live stream
    """

    def __init__(self, frame_read):
        super(VisionThread, self).__init__()
        self._stop_event = threading.Event()
        self.frame_read = frame_read

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        self.estimate_pose()

    def estimate_pose(self):
        while True:
            frame = self.frame_read.frame
            cv.imshow('View', frame)
            k = cv.waitKey(1)

            if k % 256 == 27:  # esc
                plt.close()
                break

            elif k % 256 == 32:  # space
                # get marker info
                corners, ids = marker_detector.get_marker_info()
                print("ids:", ids)

                # get pose (transformation vectors)
                rvecs, tvecs, frame = marker_detector.get_pose(corners, ids)

                # print & plot
                unit_z_m = np.array([0, 0, 1])  # unit-z vector in marker (model) frame
                if ids[0] is not None:
                    for id in ids:
                        tvec = tvecs[id]
                        rvec = rvecs[id]

                        # print
                        print("rvec:\n", rvec)
                        print("tvec:\n", tvec)

                        # < R = R_cm so that v_c = R_cm @ v_m >
                        R, _ = cv.Rodrigues(rvec)  # get rotation matrix
                        # print("Rotation Matrix:\n", R)

                        # get pose: < p_c = R @ p_m + tvec > then plot
                        posi = tvec  # posi = R @ [0, 0, 0]_m + tvec = tvec
                        dire = R @ unit_z_m  # dire: normal vector of marker in camera frame
                        for ax in axs:
                            ax.quiver(posi[0], posi[1], posi[2],
                                      dire[0], dire[1], dire[2], color='b')

                        plt.draw()  # update plot

                # draw
                aruco.drawDetectedMarkers(frame, corners)

                cv.imshow('View', frame)
                cv.waitKey(1)

            if self.stopped():
                cv.destroyAllWindows()
                break


# RUN /////////////////////////////////////////////////////////////////////////
# setting ---------------------------------------------------------------------
# connect to tello
tello = Tello()
tello.connect()  # enter SDK mode

# initiate visual stream
tello.streamon()
frame_read = tello.get_frame_read()

marker_detector = aT.MarkerDetectorTello(frame_read, aruco_dict, markerLength,
                                         cameraMatrix, distCoeffs)

# create Thread
vision_thread = VisionThread(frame_read)

# start Thread
vision_thread.start()

# main thread -----------------------------------------------------------------
# show plot
plt.show()

# wait for keyboard
keyboard.wait("esc")

# ending ----------------------------------------------------------------------
# finish Thread
vision_thread.stop()
vision_thread.join()

# END /////////////////////////////////////////////////////////////////////////
tello.streamoff()
