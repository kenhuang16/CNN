import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def non_maxima_suppression(scores, windows):
    """"""
    pass


def sliding_window(img, window_size=(64, 64), step_size=(16, 16), scale=(1.0, 1.0)):
    """Search cars in an image

    Parameters
    ----------
    img: numpy.ndarray
        Image array.
    window_size: 1x2 tuple, int
        Size of the sliding window.
    step_size: 1x2 tuple, int
        Size of the sliding step.
    scale: 1x2 tuple, float
        Scale of the original image

    Return
    ------
    windows: a list of window images with the size of "window_size"
    """
    new_x_size = np.int(img.shape[1]*scale[1])
    new_y_size = np.int(img.shape[0]*scale[0])
    img_resized = cv2.resize(img, (new_x_size, new_y_size))

    # plt.imshow(img_resized)
    # plt.show()

    windows = []
    y0_window = 0
    while (y0_window + window_size[0]) <= img_resized.shape[0]:
        x0_window = 0
        while (x0_window + window_size[0]) <= img_resized.shape[1]:

            windows.append(img_resized[y0_window:(y0_window + window_size[0]),
                                       x0_window:(x0_window + window_size[1])])

            x0_window += step_size[1]

        y0_window += step_size[0]

    return windows