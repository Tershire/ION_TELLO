"""
collect_images.py
collect images to be used for camera calibration

1st written by: Wonhee Lee
1st written on: 2023 APR 01
"""

# IMPORT //////////////////////////////////////////////////////////////////////
import CalibrationTools as cT

# SETTING /////////////////////////////////////////////////////////////////////
dir_path = r'C:\Users\leewh\Documents\Academics\Research\FR\Drone\Calibration_Data\\'
camera_name = 'laptop'
img_format = 'jpg'
NUM_IMGS = 5
TIME_INTERVAL = 3

# RUN /////////////////////////////////////////////////////////////////////////
cT.collect_images(NUM_IMGS, TIME_INTERVAL, dir_path, camera_name, img_format,
                  camera_id=0)
