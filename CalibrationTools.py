"""
CalibrationTools.py
class for Camera Calibration

1st written by: Wonhee Lee
1st written on: 2023 APR 01
guided by: https://medium.com/vacatronics/3-ways-to-calibrate-your-camera-using-opencv-and-python-395528a51615
           https://docs.opencv.org/5.x/dc/dbb/tutorial_py_calibration.html
"""

# IMPORT //////////////////////////////////////////////////////////////////////
import cv2 as cv
import numpy as np


# FUNCTION ////////////////////////////////////////////////////////////////////
# -----------------------------------------------------------------------------
def calibrate_chessboard(ROW, COL, file_names):
    # SETTING =================================================================
    # termination criteria (for cornerSubPix())
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # real world object points (x_w, y_w, 0) in unit: [chess board square]
    # initial guess
    obj_points_0 = np.zeros((ROW * COL, 3), dtype=np.float32)  # should be 32?
    obj_points_0[:, :2] = np.mgrid[0:ROW, 0:COL].T.reshape(-1, 2)

    # CHESSBOARD CORNER DETECTION =============================================
    # arrays to store {object, image} points for all images
    obj_points = []  # 3D (real world)
    img_points = []  # 2D (image)
    for file_name in file_names:
        # load image and convert to gray image
        img = cv.imread(file_name)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # find chess board corners
        ret, corners = cv.findChessboardCorners(gray, (ROW, COL), None)

        # if found, append points to arrays
        if ret:
            obj_points.append(obj_points_0)

            # refine corners
            corners_refined = cv.cornerSubPix(gray, corners, (11, 11),
                                              (-1, -1), criteria)
            img_points.append(corners_refined)

            # draw & display corners
            cv.drawChessboardCorners(img, (ROW, COL), corners_refined, ret)
            cv.imshow('img', img)
            cv.waitKey(0)

    # close windows
    # cv.destroyAllWindows()

    # CALIBRATION =============================================================
    # get camera {intrinsics, extrinsics}
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(obj_points,
                                                                     img_points,
                                                                     gray.shape[::-1],
                                                                     None, None)

    return ret, cameraMatrix, distCoeffs, rvecs, tvecs


# -----------------------------------------------------------------------------
def undistort(file_name, cameraMatrix, distCoeffs):
    img_name = file_name.split('\\')[-1].split('.')
    img = cv.imread(file_name)
    h, w = img.shape[:2]

    # get new camera matrix ---------------------------------------------------
    cameraMatrix_new, roi = cv.getOptimalNewCameraMatrix(cameraMatrix,
                                                         distCoeffs,
                                                         (w, h), alpha=1,
                                                         newImgSize=(w, h))

    dst = cv.undistort(img, cameraMatrix, distCoeffs, None, cameraMatrix_new)

    # show & save image
    cv.imshow(img_name[0] + '_calibrated.' + img_name[1], dst)
    cv.waitKey(0)

    # cv.imwrite(img_name[0] + '_calibrated.' + img_name[1], dst)

    # crop image --------------------------------------------------------------
    [x, y, w, h] = roi
    dst = dst[y:y + h, x:x + w]

    # show & save image
    cv.imshow(img_name[0] + '_calibrated' + '(cropped).' + img_name[1], dst)
    cv.waitKey(0)

    # cv.imwrite(img_name[0] + '_calibrated' + '(cropped).' + img_name[1], dst)


# FUNCTION: HELP //////////////////////////////////////////////////////////////
# -----------------------------------------------------------------------------
def save_camera_intrinsics(cameraMatrix, distCoeffs, dir_path):
    """
    save camera matrix and distortion coefficients to path
    """
    file = cv.FileStorage(dir_path, cv.FILE_STORAGE_WRITE)
    file.write('K', cameraMatrix)
    file.write('D', distCoeffs)

    file.release()


def load_camera_intrinsics(dir_path):
    """
    load camera matrix and distortion coefficients from path
    """
    file = cv.FileStorage(dir_path, cv.FILE_STORAGE_READ)

    cameraMatrix = file.getNode('K').mat()
    distCoeffs = file.getNode("D").mat()

    return cameraMatrix, distCoeffs
