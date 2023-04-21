"""
test_aruco_pose_tello.py
get real world pose of ArUco markers wrt. virtual drone CG pose,
then displays the pose on 3D quiver plot.

# HOW TO USE
0) run
1) key input: space
   it takes a photo. If ArUco markers are detected, it prints info & plots
2) key input: esc then esc
   it closes down

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
import DynamicsTools as dT
from DynamicsTools import hom, euc, E_hom

# SETTING /////////////////////////////////////////////////////////////////////
n = 0
# ArUco -----------------------------------------------------------------------
# dictionary choice
aruco_dict = aruco.DICT_4X4_50
markerLength = 17.5e-2    # [cm] marker side length

# camera ----------------------------------------------------------------------
camera_id = 0
camera_name = 'laptop'

# intrinsics
calib_file_name = r'C:\Users\leewh\Documents\Academics\Research\FR\Drone\Calibration_Data\\' + \
                  f'{camera_name}_intrinsics.yml'
cameraMatrix, distCoeffs = cT.load_camera_intrinsics(calib_file_name)

# drone properties ------------------------------------------------------------
l = 3.8e-2    # distance camera to drone CG
CAMERA_TILT = +7    # [deg] camera tilt angle (positive(+) when looking down)

# frame relationship ----------------------------------------------------------
"""
NOTATION: p_a_b: p of a in b frame
ex) o_b_b: origin of body in body frame
"""
o_b_b = np.array([0, 0, 0])    # origin of drone
i_b_b = np.array([1, 0, 0])    # x unit vector of drone

R_bc = dT.R_z(+90, 'deg') @ dT.R_x(+90 + CAMERA_TILT, 'deg')
# R_bc = dT.R_x(+90, 'deg') @ dT.R_y(-90, 'deg')
t_bc = l * +i_b_b

E_hom_bc = E_hom(R_bc, t_bc)    # Euclidean frame transform from c to b

# frame conversion ------------------------------------------------------------
o_c_c = np.array([0, 0, 0])
o_c_b = euc(E_hom_bc @ hom(o_c_c))
print(o_c_b)

k_c_c = np.array([0, 0, 1])
k_c_b = R_bc @ k_c_c
print(k_c_b)

# plot ------------------------------------------------------------------------
fig = plt.figure(figsize=plt.figaspect(1 / 3))
axs = []
for i in range(3):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    axs.append(ax)

# plot drone pose
for ax in axs:
    ax.quiver(o_b_b[0], o_b_b[1], o_b_b[2],
              i_b_b[0], i_b_b[1], i_b_b[2], color='k')

# plot camera pose
for ax in axs:
    ax.quiver(o_c_b[0], o_c_b[1], o_c_b[2],
              k_c_b[0], k_c_b[1], k_c_b[2], color='y')

# setting
for ax in axs:
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.5, +0.5)
    ax.set_zlim(-0.5, +0.5)

    ax.set_xlabel('x_{c}')
    ax.set_ylabel('y_{c}')
    ax.set_zlabel('z_{c}')

# marker as myself
# axs[0].view_init(elev=+90, azim=-90, roll=-90)
# axs[1].view_init(elev=0, azim=-90, roll=0)
# axs[2].view_init(elev=0, azim=0, roll=0)

# drone as myself
axs[0].view_init(elev=+90, azim=-90, roll=+90)
axs[1].view_init(elev=0, azim=-90, roll=0)
axs[2].view_init(elev=0, azim=180, roll=0)


# CLASS ///////////////////////////////////////////////////////////////////////
class VisionThread(threading.Thread):
    """
    "Killable Thread" to show live stream
    """

    def __init__(self, cap):
        super(VisionThread, self).__init__()
        self._stop_event = threading.Event()
        self.cap = cap

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        self.estimate_pose()

    def estimate_pose(self):
        global n

        while True:
            ret, frame = self.cap.read()
            cv.imshow('View', frame)
            k = cv.waitKey(1)

            print(n)

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
                o_m_m = np.array([0, 0, 0])
                k_m_m = np.array([0, 0, 1])
                if ids[0] is not None:
                    for id in ids:
                        r_cm = rvecs[id]
                        t_cm = tvecs[id]

                        R_cm, _ = cv.Rodrigues(r_cm)

                        # frame transform -------------------------------------
                        # position
                        o_m_b = euc(E_hom_bc @ E_hom(R_cm, t_cm) @ hom(o_m_m))

                        # orientation
                        k_m_b = R_bc @ R_cm @ k_m_m

                        # scale
                        k_m_b_scaled = k_m_b * 0.5

                        for ax in axs:
                            ax.quiver(o_m_b[0], o_m_b[1], o_m_b[2],
                                      k_m_b_scaled[0], k_m_b_scaled[1], k_m_b_scaled[2], color='b')

                        print("o_m_b:\n", o_m_b)

                        plt.draw()  # update plot

                # draw
                aruco.drawDetectedMarkers(frame, corners)

                cv.imshow('View', frame)
                cv.waitKey(1)

            if self.stopped():
                cap.release()
                cv.destroyAllWindows()
                break

# -----------------------------------------------------------------------------
class TestThread(threading.Thread):
    """
    "Killable Thread" to show live stream
    """

    def __init__(self):
        super(TestThread, self).__init__()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        self.func()

    def func(self):
        global n

        while True:
            n += 1
            print(n)


# RUN /////////////////////////////////////////////////////////////////////////
# setting ---------------------------------------------------------------------
# create object
cap = cv.VideoCapture(camera_id)

marker_detector = aT.MarkerDetector(cap, aruco_dict, markerLength,
                                   cameraMatrix, distCoeffs)

# create Thread
vision_thread = VisionThread(cap)
testThread = TestThread()

# start Thread
vision_thread.start()
testThread.start()

# main thread -----------------------------------------------------------------
# show plot
plt.show()

# wait for keyboard
# while True:
#     n += 1
#     print("main", n)
keyboard.wait("esc")

# ending ----------------------------------------------------------------------
# finish Thread
vision_thread.stop()
vision_thread.join()

testThread.stop()
testThread.join()
