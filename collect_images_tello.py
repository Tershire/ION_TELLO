"""
collect_images_tello.py
collect images to be used for camera calibration for TELLO Edu drone

1st written by: Wonhee Lee
1st written on: 2023 APR 01
"""

# IMPORT //////////////////////////////////////////////////////////////////////
import CalibrationTools as cT
from djitellopy import Tello

# CONNECT TO TELLO ////////////////////////////////////////////////////////////
tello = Tello()
tello.connect()  # enter SDK mode

# VISION SETUP ////////////////////////////////////////////////////////////////
# initiate visual stream
tello.streamon()
frame_read = tello.get_frame_read()

# SETTING /////////////////////////////////////////////////////////////////////
dir_path = r'C:\Users\leewh\Documents\Academics\Research\FR\Drone\Calibration_Data\\'
camera_name = 'tello'
img_format = 'jpg'
NUM_IMGS = 35
TIME_INTERVAL = 3

# RUN /////////////////////////////////////////////////////////////////////////
cT.collect_images_tello(frame_read, NUM_IMGS, TIME_INTERVAL, dir_path,
                        camera_name, img_format)

# END /////////////////////////////////////////////////////////////////////////
tello.streamoff()
