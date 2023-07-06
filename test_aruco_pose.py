"""
test_aruco_pose.py
get real world pose of ArUco markers wrt. camera pose,
then displays the pose on 3D quiver plot.

# HOW TO USE
0) run
1) key input: space
   it takes a photo. If ArUco markers are detected, it prints info & plots
2) key input: esc then esc
   it closes down

1st written by: Wonhee Lee
1st written on: 2023 MAR 30
referred: https://stackoverflow.com/questions/34588464/python-how-to-capture-image-from-webcam-on-click-using-opencv
          https://stackoverflow.com/questions/19329039/plotting-animated-quivers-in-python
          https://matplotlib.org/stable/gallery/mplot3d/subplot3d.html
          https://amroamroamro.github.io/mexopencv/matlab/cv.solvePnP.html
          https://stackoverflow.com/questions/31004394/threading-script-not-printing-to-console
          https://stackoverflow.com/questions/63460614/why-is-the-printing-messed-up-when-running-multiple-threads-in-python
          https://stackoverflow.com/questions/23100704/running-infinite-loops-using-threads-in-python
          https://stackoverflow.com/questions/37123486/thread-starvation-while-locking-in-a-loop-in-python
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
import sys

# GLOBAL VARIABLE /////////////////////////////////////////////////////////////
# control & guidance ----------------------------------------------------------
o_t_b = None  # target position    in drone body frame
k_t_b = None  # target orientation in drone body frame
id_to_follow = 3  # id  of ArUco marker to follow
ids = []  # ids of detected ArUco markers
speed = 10  # drone flight speed [cm/s] (limit: [10, 100])
l = 45e-2  # gap between marker and sensor [m]

# GLOBAL CONSTANT /////////////////////////////////////////////////////////////
# drone properties ------------------------------------------------------------
L = 3.8e-2  # distance camera to drone CG
CAMERA_TILT = +7  # [deg] camera tilt angle (positive(+) when looking down)

# control & guidance ----------------------------------------------------------
# control gain
K_X = 10
K_Y = 25
K_Z = 25
K_psi = 15

# safety limit ----------------------------------------------------------------
MIN_HEIGHT = 35  # minimum flight height [cm]

# SETTING /////////////////////////////////////////////////////////////////////
# ArUco -----------------------------------------------------------------------
# dictionary choice
aruco_dict = aruco.DICT_4X4_50
markerLength = 17.5e-2  # [cm] marker side length

# camera ----------------------------------------------------------------------
camera_id = 0
camera_name = 'laptop'

# intrinsics
calib_file_name = r'C:\Users\leewh\Documents\Academics\Research\FR\Drone\Calibration_Data\\' + \
                  f'{camera_name}_intrinsics.yml'
cameraMatrix, distCoeffs = cT.load_camera_intrinsics(calib_file_name)

# threading -------------------------------------------------------------------
lock = threading.Lock()

# frame relationship ----------------------------------------------------------
o_b_b = np.array([0, 0, 0])  # origin of drone  in b
i_b_b = np.array([1, 0, 0])  # normal of drone  in b
o_c_c = np.array([0, 0, 0])  # origin of camera in c
k_c_c = np.array([0, 0, 1])  # normal of camera in c
o_m_m = np.array([0, 0, 0])  # origin of marker in m
k_m_m = np.array([0, 0, 1])  # normal of marker in m
o_t_t = np.array([0, 0, 0])  # origin of target in t
k_t_t = np.array([0, 0, 1])  # normal of target in t

# transform: camera to drone body
R_bc = dT.R_z(+90, 'deg') @ dT.R_x(+90 + CAMERA_TILT, 'deg')
t_bc = L * +i_b_b
E_hom_bc = E_hom(R_bc, t_bc)  # Euclidean frame transform from c to b

o_c_b = euc(E_hom_bc @ hom(o_c_c))
k_c_b = R_bc @ k_c_c
print('o_c_b:', o_c_b)
print('k_c_b:', k_c_b)

# transform: target to marker
R_mt = np.eye(3)
t_mt = l * +k_m_m
E_hom_mt = E_hom(R_mt, t_mt)  # Euclidean frame transform from t to m

o_t_m = euc(E_hom_mt @ hom(o_t_t))
print('o_t_m:', o_t_m)

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

    ax.set_xlabel('x_{b}')
    ax.set_ylabel('y_{b}')
    ax.set_zlabel('z_{b}')

# marker as myself
# axs[0].view_init(elev=+90, azim=-90, roll=-90)
# axs[1].view_init(elev=0, azim=-90, roll=0)
# axs[2].view_init(elev=0, azim=0, roll=0)

# drone as myself
axs[0].view_init(elev=+90, azim=-90, roll=+90)
axs[1].view_init(elev=0, azim=-90, roll=0)
# axs[2].view_init(elev=0, azim=180, roll=0)


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
        """
        get estimated marker pose in drone frame
        """
        global o_t_b, k_t_b, ids

        while True:
            ret, frame = self.cap.read()
            cv.imshow('View', frame)
            k = cv.waitKey(1)

            if k % 256 == 27:  # esc
                plt.close()
                break

            elif k % 256 == 32:  # space
                # get marker info
                corners, ids = marker_detector.get_marker_info()
                # print("ids:", ids)

                # get pose (transformation vectors)
                rvecs, tvecs, frame = marker_detector.get_pose(corners, ids)

                if ids[0] is not None:
                    for id in ids:
                        r_cm = rvecs[id]
                        t_cm = tvecs[id]

                        R_cm, _ = cv.Rodrigues(r_cm)

                        # frame transform -------------------------------------
                        # position
                        o_t_b = euc(E_hom_bc @ E_hom(R_cm, t_cm) @ hom(o_t_m))
                        # o_m_b = euc(E_hom_bc @ E_hom(R_cm, t_cm) @ hom(o_m_m))
                        print("o_t_b: ", o_t_b)

                        # orientation
                        k_t_b = R_bc @ R_cm @ R_mt @ k_t_t
                        # k_m_b = R_bc @ R_cm @ k_m_m
                        print("k_t_b: ", k_t_b)

                        for ax in axs:
                            ax.quiver(o_t_b[0], o_t_b[1], o_t_b[2],
                                      k_t_b[0], k_t_b[1], k_t_b[2], color='b')

                        plt.draw()  # update plot

                        follow()

                    # draw
                    aruco.drawDetectedMarkers(frame, corners)

                    cv.imshow('View', frame)
                    cv.waitKey(1)

                else:  # if marker to follow is lost, stay at the current location
                    o_t_b = None
                    k_t_b = None

            # termination condition -------------------------------------------
            if self.stopped() or keyboard.is_pressed('esc'):
                cv.destroyAllWindows()
                break


# control ---------------------------------------------------------------------
def follow():
    """
    follows marker by yawing & translating
    """
    global o_t_b, k_t_b, ids

    # k = cv.waitKey(1)
    # if k % 256 == 32:
    
    if id_to_follow in ids and \
            o_t_b is not None and \
            k_t_b is not None:

        # rotate
        psi = get_delta_yaw()

        if psi > 0:
            # tello.rotate_counter_clockwise(int(np.rad2deg(psi)))
            with lock:
                print("psi [deg]:", np.rad2deg(psi), flush=True)
                sys.stdout.flush()
        elif psi < 0:
            # tello.rotate_clockwise(int(np.rad2deg(abs(psi))))
            with lock:
                print("psi [deg]:", np.rad2deg(psi), flush=True)
                sys.stdout.flush()
        else:
            pass

        # translate
        # o_t_b_in_cm = o_t_b * 100
        # tello.go_xyz_speed(int(o_t_b_in_cm[0]),
        #                    int(o_t_b_in_cm[1]),
        #                    int(o_t_b_in_cm[2]),
        #                    int(speed))

        # RC command method
        # tello.send_rc_control(int(K_X * o_t_b[0]),
        #                       int(K_Y * o_t_b[1]),
        #                       int(K_Z * o_t_b[2]),
        #                       int(K_psi * psi))

        print("rc command: ", round(K_X * o_t_b[0]),
                              round(K_Y * o_t_b[1]),
                              round(K_Z * o_t_b[2]),
                              round(K_psi * psi))

        # with lock:
        #     print("o_t_b:", o_t_b, flush=True)
        #     sys.stdout.flush()
        #     print("k_t_b:", k_t_b, flush=True)
        #     sys.stdout.flush()


def get_delta_yaw():
    """
    calculate delta yaw to align drone x axis and
    target z axis projected onto drone xy plane
    """
    global o_t_b, k_t_b

    if o_t_b is not None and \
       k_t_b is not None:
        k_t_b_proj = k_t_b[0:2]  # normal of target projected on {x_b, y_b} plane
        if k_t_b_proj[0] < 0:  # if drone is confronting marker
            psi = np.arccos(np.dot(-k_t_b_proj / np.linalg.norm(k_t_b_proj), i_b_b[0:2]))

            # set yaw direction because psi is [0, pi]
            if k_t_b_proj[1] > 0:
                psi = -psi
            elif k_t_b_proj[1] < 0:
                pass
        else:
            psi = 0

        return psi


# RUN /////////////////////////////////////////////////////////////////////////
# setting ---------------------------------------------------------------------
# create object
cap = cv.VideoCapture(camera_id)

marker_detector = aT.MarkerDetector(cap, aruco_dict, markerLength,
                                    cameraMatrix, distCoeffs)

# create Thread
vision_thread = VisionThread(cap)

# start Thread
vision_thread.start()

# main thread -----------------------------------------------------------------
# show plot
plt.show()

# while True:
#     follow()
#
#     if keyboard.is_pressed('esc'):  # esc
#         print("Flight Ended by the User. Landing...")
#         break

keyboard.wait("esc")
print("Flight Ended by the User. Landing...")

# ending ----------------------------------------------------------------------
# finish Thread
vision_thread.stop()
vision_thread.join()
