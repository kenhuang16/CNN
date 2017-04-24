#!/usr/bin/python
"""
Calibrate and apply (inverse) perspective transform.
"""
import os
import numpy as np
import pickle
import cv2
from scipy import misc

from my_plot import double_plot


def calibrate_perspective_trans(src, dst, output="perspective_trans.pkl"):
    """Get the transform and inverse transform matrix of the
       perspective transform.

    Parameters
    ----------
    src: numpy.ndarray()
        Source points.
    dst: numpy.ndarray()
        Object points.
    output: string
        Pickle file to store the perspective transform parameters.
    """
    p_matrix = cv2.getPerspectiveTransform(src, dst)
    inv_p_matrix = cv2.getPerspectiveTransform(dst, src)

    perspective_trans = dict()
    perspective_trans['matrix'] = p_matrix
    perspective_trans['inv_matrix'] = inv_p_matrix
    with open(output, 'wb') as fp:
        pickle.dump(perspective_trans, fp)
    print("Perspective transform matrices were saved in {}.".format(output))


def perspective_trans(img, filename, inverse=False):
    """Apply a perspective transform.

    Parameters
    ----------
    img: numpy.ndarray()
        Image array.
    filename: string
        File stores the transform matrices.
    inverse: Boolean
        True for inverse transform.
    """
    with open(filename, "rb") as fp:
        perspective_trans = pickle.load(fp)
    trans_matrix = perspective_trans["matrix"]
    inverse_trans_matrix = perspective_trans["inv_matrix"]

    if inverse is False:
        return cv2.warpPerspective(img, trans_matrix, img.shape[:2][::-1])
    else:
        return cv2.warpPerspective(img, inverse_trans_matrix, img.shape[:2][::-1])


if __name__ == "__main__":
    test_img = './test_images/test_image_perspective_trans.png'

    img = misc.imread(test_img)

    src = np.float32([[0, 720], [570, 450], [708, 450], [1280, 720]])
    dst = np.float32([[0, 720], [0, 0], [1280, 0], [1280, 720]])

    perspective_trans_file = "perspective_trans.pkl"
    if not os.path.isfile(perspective_trans_file):
        calibrate_perspective_trans(src, dst, output=perspective_trans_file)

    # Apply perspective transform on a test image
    warped = perspective_trans(img, perspective_trans_file)

    # Visualize the transform
    cv2.polylines(img, np.int32([src]), 1, (255, 255, 0), thickness=4)
    cv2.polylines(warped, np.int32([dst]), 1, (255, 255, 0), thickness=4)
    double_plot(img, warped,
                ('original', 'warped', 'perspective transform'), output='')