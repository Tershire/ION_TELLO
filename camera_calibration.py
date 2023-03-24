"""
camera_calibration.py
calibrate the grid image distortion

<!><A> ASSUMPTIONS <A><!>
object points are assumed to be on plane (z position = 0)

1st written by: Wonhee Lee
1st written on: 2023 JAN 26
guided by: https://docs.opencv.org/5.x/dc/dbb/tutorial_py_calibration.html

"""

# IMPORT //////////////////////////////////////////////////////////////////////
import cv2 as cv
import numpy as np
import glob


# SETTING /////////////////////////////////////////////////////////////////////
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # yet

# chess board inner grid pattern size (row, col)
ROW = 9
COL = 6

# real world object points (3D) in unit: [chess board square] 
obj_points_0        = np.zeros((ROW * COL, 3), dtype = np.float32) # should be 32? not 64?
obj_points_0[:, :2] = np.mgrid[0:ROW, 0:COL].T.reshape(-1, 2)

# load all image file names
file_names = glob.glob(r'C:\opencv\sources\samples\data\right0*.jpg')
# file_names = glob.glob(r'C:\Users\leewh\Documents\Academics\EXC Learning\Computer Vision\OpenCV-Python Tutorials\chessboard03.jpg')

# test for one img
# file_names = [file_names[0]]


# CHESSBOARD CORNER DETECTION /////////////////////////////////////////////////
# arrays to store {object, image} points for all images
obj_points = []; # 3D (real world)
img_points = []; # 2D (image)

for file_name in file_names:
    # load image and convert to gray image
    img  = cv.imread(file_name)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # find chess board corners ("why corners is in shape (54, 1, 2)?")
    [ret, corners] = cv.findChessboardCorners(gray, (ROW, COL), None)
    # * ret [bool]: True if found
    
    # if found, append points to arrays
    if ret == True:
        obj_points.append(obj_points_0)

        # refine corners ("are the parameters ok?")
        corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                          criteria)     
        img_points.append(corners_refined)
        
        # draw & display corners
        cv.drawChessboardCorners(img, (ROW, COL), corners_refined, ret)
        cv.imshow('img', img)
        cv.waitKey(0)

# close windows        
# cv.destroyAllWindows()
    

# CALIBRATION /////////////////////////////////////////////////////////////////
# get camera matrix, distortion coefficients, {rotation, translation) vectors
# ("are the parameters ok? why [::-1]")
[ret, cam_matrix, disto_Cs, rvecs, tvecs] = cv.calibrateCamera(obj_points,
                                                              img_points,
                                                              gray.shape[::-1],
                                                              None, None)


# DISTORTION CORRECTION ///////////////////////////////////////////////////////
# pick an image to calibrate (get one arbitrary image that is used)
file_name = file_names[0]
img_name = file_name.split('\\')[-1].split('.')
img = cv.imread(file_name)

# refine camera matrix
alpha = 1 #"what does this parameter do?"
[h, w] = img.shape[:2]
[cam_matrix_refined, roi] = cv.getOptimalNewCameraMatrix(cam_matrix, disto_Cs, 
                                                     (w, h), alpha, (w, h))

# undistort
dst = cv.undistort(img, cam_matrix, disto_Cs, None, cam_matrix_refined)

# show & save image
cv.imshow(img_name[0] + '_calibrated.' + img_name[1], dst)
cv.waitKey(0)

# cv.imwrite(img_name[0] + '_calibrated.' + img_name[1], dst)

# crop image
[x, y, w, h] = roi
dst = dst[y:y+h, x:x+w]

# show & save image
cv.imshow(img_name[0] + '_calibrated' + '(cropped).' + img_name[1], dst)
cv.waitKey(0)

# cv.imwrite(img_name[0] + '_calibrated' + '(cropped).' + img_name[1], dst)
