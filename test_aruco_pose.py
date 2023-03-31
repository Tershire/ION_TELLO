"""
test_aruco_pose.py
get real world pose of ArUco markers wrt. camera pose,
then displays the pose on 3D quiver plot.

1st written by: Wonhee Lee
1st written on: 2023 MAR 30
referred: https://stackoverflow.com/questions/34588464/python-how-to-capture-image-from-webcam-on-click-using-opencv
          https://stackoverflow.com/questions/19329039/plotting-animated-quivers-in-python
          https://matplotlib.org/stable/gallery/mplot3d/subplot3d.html
"""

# IMPORT //////////////////////////////////////////////////////////////////////
import cv2 as cv
import numpy as np
import threading
import keyboard
import ArUcoTools as a_t
import matplotlib.pyplot as plt


# SETTING /////////////////////////////////////////////////////////////////////
# ArUco -----------------------------------------------------------------------
# dictionary choice
aruco_dict = cv.aruco.DICT_4X4_50
markerLength = 0.175

# camera ----------------------------------------------------------------------
camera_id = 0

# temporary trials: actual calibration process must be implemented
cameraMatrix = np.array([[543.05833681, 0., 326.0951866],
                         [0., 542.67378833, 247.65515938],
                         [0., 0., 1.]])
distCoeffs = np.array([-0.28608759, 0.13647301, -0.00076189, 0.0014116, -0.06865808])


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

            if k % 256 == 27:  # esc
                plt.close()
                break

            elif k % 256 == 32:  # space
                # get marker info
                corners, ids = marker_tracker.get_marker_info()
                print("ids:", ids)

                # get pose (transformation vectors)
                rvecs, tvecs, frame = marker_tracker.get_pose(corners, ids)

                # print & plot
                unit_z_m = np.array([0, 0, 1])    # unit-z vector in marker (model) frame
                if ids[0] is not None:
                    for id in ids:
                        tvec = tvecs[id]
                        rvec = rvecs[id]

                        # print
                        print("rvec:\n", rvec)
                        print("tvec:\n", tvec)

                        # < R = R_cm so that v_c = R_cm @ v_m >
                        R, _ = cv.Rodrigues(rvec)    # get rotation matrix
                        # print("Rotation Matrix:\n", R)

                        # get pose: < p_c = R @ p_m + tvec > then plot
                        posi = tvec            # posi = R @ [0, 0, 0]_m + tvec = tvec
                        dire = R @ unit_z_m    # dire: normal vector of marker in camera frame
                        for ax in axs:
                            ax.quiver(posi[0], posi[1], posi[2],
                                      dire[0], dire[1], dire[2], color='b')

                        plt.draw()    # update plot

                # draw
                cv.aruco.drawDetectedMarkers(frame, corners)

                cv.imshow('View', frame)
                cv.waitKey(1)

            if self.stopped():
                cap.release()
                cv.destroyAllWindows()
                break


# EVENT LOOP //////////////////////////////////////////////////////////////////
# setting ---------------------------------------------------------------------
# create object
cap = cv.VideoCapture(camera_id)

marker_tracker = a_t.MarkerTracker(cap, aruco_dict, markerLength,
                                   cameraMatrix, distCoeffs)

# create Thread
streamer = ArUcoStreamer(cap)

# start Thread
streamer.start()

# main thread -----------------------------------------------------------------
# show plot
plt.show()

# wait for keyboard
keyboard.wait("esc")

# ending ----------------------------------------------------------------------
# finish Thread
streamer.stop()
streamer.join()
