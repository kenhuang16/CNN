#!/usr/bin/python
"""
Calibrate camera and undistort images.
"""
import numpy as np
import cv2
import glob
import pickle

from utilities import two_plots


def calibrate_camera(chess_board_images, pattern_size=(None, None),
                     output="camera_cali.pkl", show_chess_board=True):
    """Calibrate camera

    :param chess_board_images: list of strings
        List of chess board file names.
    :param pattern_size: tuple, 2x1
        Number of inner corners per a chessboard row and column.
    :param output: string
        Pickle file for storing the result.
    :param show_chess_board: Boolean
        True for showing each result of finding corners.
    """
    pts_row = pattern_size[0]
    pts_col = pattern_size[1]

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pts_col*pts_row, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pts_row, 0:pts_col].T.reshape(-1, 2)  # x and y

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d points in real world space
    img_points = []  # 2d points in image plane

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(chess_board_images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        retval, corners = cv2.findChessboardCorners(gray, (pts_row, pts_col), None)
        print('{}: {}'.format(fname, retval))

        if retval is True:
            obj_points.append(objp)
            img_points.append(corners)

            if show_chess_board is True:
                cv2.drawChessboardCorners(img, (pts_row, pts_col), corners, retval)
                cv2.imshow('img', img)
                cv2.waitKey(500)

    if show_chess_board is True:
        cv2.destroyAllWindows()

    camera_cali = dict()
    camera_cali["obj_points"] = obj_points
    camera_cali["img_points"] = img_points
    with open(output, "wb") as f:
        pickle.dump(camera_cali, f)
    print("Calibration of the camera was saved in {}.".format(output))


def undistort_image(img, obj_points, img_points):
    """Undistort an image

    :param img: numpy.ndarray
        Original image.
    :param obj_points: List of numpy.ndarray. Each array has the shape
                       (N, 3), where N is the number of points.
        List of 3d points in the real world space of each image.
    :param img_points: List of numpy.ndarray. Each array has the shape
                       (N, 1, 2), where N is the number of points.
        List of 2d points in the image plane of each each image

    :return undistorted: numpy.ndarray
        Undistorted image.
    """
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = \
        cv2.calibrateCamera(obj_points, img_points, img.shape[:2],
                            None, None)

    undistorted = cv2.undistort(
        img, camera_matrix, dist_coeffs, None, camera_matrix)

    return undistorted
