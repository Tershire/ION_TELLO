"""
undistort.py
undistort

1st written by: Wonhee Lee
1st written on: 2023 JUL 06
"""

# IMPORT //////////////////////////////////////////////////////////////////////
import CalibrationTools as cT
import glob


# GLOBAL VARIABLE & CONSTANT //////////////////////////////////////////////////
save_data = False;


# SETTING /////////////////////////////////////////////////////////////////////
# load all image file
dir_path = r'/home/tershire/Documents/SLAM_dataset/KissFly/ovoide_1_test/cam0/'
camera_name = 'Intel_T265'    # <!> BE SURE TO CHANGE NOT TO LOSE SAVED DATA <!>
img_format = 'png'
file_names = glob.glob(dir_path + '*' +
                       '.' + f'{img_format}')

cameraMatrix, distCoeffs = cT.load_camera_intrinsics(r'/home/tershire/Documents/SLAM_dataset/KissFly/calibration_data/' + \
                                                     f'{camera_name}_intrinsics.yml')

print(cameraMatrix, distCoeffs)

# RUN /////////////////////////////////////////////////////////////////////////
# test: undistort
cT.undistort(file_names[0], cameraMatrix, distCoeffs, False)

