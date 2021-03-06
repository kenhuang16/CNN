"""
Apply gradient and/or color threshold to an image.
"""
import numpy as np
import cv2

from utilities import change_colorspace


class Threshold(object):
    """Theshold class"""
    def __init__(self, img, color_space, channel):
        """Initialization

        :param img: numpy.ndarray
            Original image.
            The original image is assumed to have the RGB color space.
        :param color_space: string
            Color space of the new image.
        :param channel: int
            Index of the channel. Must be None, 0, 1 or 2.
        """
        new_img = change_colorspace(img, color_space)

        if len(new_img.shape) == 2:
            self.img = new_img
        elif len(new_img.shape) == 3:
            assert (channel in range(3))
            self.img = new_img[:, :, channel]
        else:
            raise ValueError()

        # initialize the binary image
        self.binary = np.zeros(img.shape[:2]).astype(np.uint8)

    def transform(self, direction=None, **kwargs):
        """Apply threshold operation

        :param direction: string
            Name of the direction in gradient calculation.
        """
        if direction is None:
            self.color_thresh(**kwargs)
        elif direction in ('mag', 'x', 'y', 'angle'):
            self.gradient_thresh(direction, **kwargs)
        else:
            raise ValueError('Unknown value for direction')

    def gradient_thresh(self, direction, thresh=(0, 255), sobel_kernel=3):
        """Apply gradient threshold to a channel of an image

        :param direction: string
            Name of the direction in gradient calculation.
        :param thresh: tuple, (min, max)
            Threshold values.
        :param sobel_kernel: int
            Sober kernel size.
        """
        abs_sobelx = np.abs(
            cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        abs_sobely = np.abs(
            cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

        if direction == 'mag':
            value = np.sqrt(abs_sobelx ** 2 + abs_sobely ** 2)
        elif direction == 'x':
            value = abs_sobelx
        elif direction == 'y':
            value = abs_sobely
        elif direction == 'angle':
            value = np.arctan2(abs_sobely, abs_sobelx)
        else:
            raise ValueError('Unknown gradient type!')

        scaled_sobel = np.uint8(255 * value / np.max(value))

        self.binary = np.zeros_like(self.binary)
        self.binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    def color_thresh(self, thresh=(0, 255)):
        """Apply color threshold to a channel of an image

        :param thresh: tuple, (min, max)
            Threshold values.
        """
        self.binary = np.zeros_like(self.binary)
        self.binary[(self.img > thresh[0]) & (self.img <= thresh[1])] = 1


def thresh_image(img, params):
    """Apply the combination of different thresholds

    :param img: numpy.ndarray
        Original image.
    :param params: dictionary
        A list of parameters used for threshing
        e.g.
        [{'color_space': 'hls', 'channel': 2, 'direction': 'x', 'thresh': (20, 100)},
         {'color_space': 'hls', 'channel': 2, 'direction': None, 'thresh': (100, 255)},

    :return binary: numpy.ndarray
        Binary image after applying threshold.
    """
    # Apply gradient and color threshold
    binary = None
    for param in params:
        th = Threshold(img, param['color_space'], param['channel'])

        th.transform(param['direction'], thresh=param['thresh'])
        if binary is None:
            binary = th.binary
        else:
            binary |= th.binary

    return binary
