import cv2
import numpy as np


def change_colorspace(img, color_space):
    """Convert the color space of an image

    Parameters
    ----------
    img: numpy.ndarray
        Image array.

    Returns
    -------
    New image array.
    """
    if len(img.shape) < 3:
        raise ValueError("A color image is required!")

    # apply color conversion if other than 'RGB'
    if color_space == 'GRAY':
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif color_space == 'RGB':
        new_img = np.copy(img)
    elif color_space == 'HSV':
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif color_space == 'LUV':
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    elif color_space == 'HLS':
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif color_space == 'YUV':
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    elif color_space == 'YCrCb':
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else:
        raise ValueError("Unknown color space!")

    return new_img

