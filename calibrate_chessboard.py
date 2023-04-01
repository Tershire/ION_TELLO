"""
calibrate_chessboard.py
calibrate camera using chessboard

1st written by: Wonhee Lee
1st written on: 2023 APR 01
"""

# IMPORT //////////////////////////////////////////////////////////////////////
import cv2 as cv
import CalibrationTools as c_t
import glob

# SETTING /////////////////////////////////////////////////////////////////////
# chess board inner grid pattern size (row, col)
ROW = 9
COL = 6

# load all image file names
file_names = glob.glob(r'C:\opencv\sources\samples\data\right0*.jpg')

# test for one img
# file_names = [file_names[0]]


# RUN /////////////////////////////////////////////////////////////////////////
ret, cameraMatrix, distCoeffs, rvecs, tvecs = c_t.calibrate_chessboard(ROW, COL, file_names)

print(cameraMatrix)
print(distCoeffs)

# save data
dir_path = r'C:\Users\leewh\Documents\Academics\Research\FR\Drone\Calibration_Data\laptop_camera_intrinsics.yml'
c_t.save_camera_intrinsics(cameraMatrix, distCoeffs, dir_path)