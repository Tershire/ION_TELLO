"""
calibrate_chessboard.py
calibrate camera using chessboard

1st written by: Wonhee Lee
1st written on: 2023 APR 01
"""

# IMPORT //////////////////////////////////////////////////////////////////////
import CalibrationTools as cT
import glob

# SETTING /////////////////////////////////////////////////////////////////////
# calibration -----------------------------------------------------------------
# chess board inner grid pattern size (row, col)
ROW = 9
COL = 6

# load all image file
dir_path = r'C:\Users\leewh\Documents\Academics\Research\FR\Drone\Calibration_Data\\'
camera_name = 'laptop'    # <!> BE SURE TO CHANGE NOT TO LOSE SAVED DATA <!>
img_format = 'jpg'
file_names = glob.glob(dir_path + camera_name + '_*' +
                       '.' + f'{img_format}')

# test for one img
# file_names = [file_names[0]]

# data output -----------------------------------------------------------------
save_data = False

# RUN /////////////////////////////////////////////////////////////////////////
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cT.calibrate_chessboard(ROW, COL, file_names)

print(cameraMatrix)
print(distCoeffs)

# test: undistort
cT.undistort(file_names[0], cameraMatrix, distCoeffs, False)

# save data
if save_data:
    calib_file_name = r'C:\Users\leewh\Documents\Academics\Research\FR\Drone\Calibration_Data\\' + \
                      f'{camera_name}_intrinsics.yml'
    cT.save_camera_intrinsics(cameraMatrix, distCoeffs, calib_file_name)
