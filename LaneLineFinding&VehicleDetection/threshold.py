#!/usr/bin/python
"""
Apply gradient and/or color threshold to an image.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utilities import two_plots


DEBUG = False


class Threshold(object):
    """Theshold class"""
    def __init__(self, img):
        """Initialization

        :param img: numpy.ndarray
            Original image.
        """
        self.img = img
        self.channel_ = None
        self.binary = np.zeros(img.shape[:2]).astype(np.uint8)

    def transform(self, channel, direct=None, **kwargs):
        """Apply threshold operation

        :param channel: string
            Name of the color channel.
        :param direct: string
            Name of the direction in gradient calculation.
        """

        if direct is None:
            binary = self.color_thresh(channel, **kwargs)
        elif direct in ('mag', 'x', 'y', 'angle'):
            binary = self.gradient_thresh(channel, direct, **kwargs)
        else:
            raise ValueError('Unknown value for direct')

        self.binary = self.binary | binary

    def gradient_thresh(self, channel, direct, thresh=(0, 255), sobel_kernel=3):
        """Apply gradient threshold to a channel of an image

        :param channel: string
            Name of the color channel.
        :param direct: string
            Name of the direction in gradient calculation.
        :param thresh: tuple, (min, max)
            Threshold values.
        :param sobel_kernel: int
            Sober kernel size.

        :return: binary: numpy.ndarray
            Binary image.
        """
        if channel == 'gray':
            self.channel_ = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        else:
            hls = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS)

            if channel == 'h':
                self.channel_ = hls[:, :, 0]
            elif channel == 'l':
                self.channel_ = hls[:, :, 1]
            elif channel == 's':
                self.channel_ = hls[:, :, 2]
            else:
                raise ValueError("Unknown channel!")

        abs_sobelx = np.abs(
            cv2.Sobel(self.channel_, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        abs_sobely = np.abs(
            cv2.Sobel(self.channel_, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

        if direct == 'mag':
            value = np.sqrt(abs_sobelx ** 2 + abs_sobely ** 2)
        elif direct == 'x':
            value = abs_sobelx
        elif direct == 'y':
            value = abs_sobely
        elif direct == 'angle':
            value = np.arctan2(abs_sobely, abs_sobelx)
        else:
            raise ValueError('Unknown gradient type!')

        scaled_sobel = np.uint8(255 * value / np.max(value))

        binary = np.zeros_like(self.binary)
        binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return binary

    def color_thresh(self, channel, thresh=(0, 255)):
        """Apply color threshold to a channel of an image

        :param channel: string
            Name of the color channel.
        :param thresh: tuple, (min, max)
            Threshold values.

        :return: binary: numpy.ndarray
            Binary image.
        """
        if channel == 'gray':
            self.channel_ = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        else:
            hls = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS)
            if channel == 'h':
                self.channel_ = hls[:, :, 0]
            elif channel == 'l':
                self.channel_ = hls[:, :, 1]
            elif channel == 's':
                self.channel_ = hls[:, :, 2]
            else:
                raise ValueError("Unknown channel!")

        binary = np.zeros_like(self.binary)
        binary[(self.channel_ > thresh[0]) & (self.channel_ <= thresh[1])] = 1

        return binary


if __name__ == "__main__":
    test_image = "./output_images_P4/threshold_original.jpg"

    thresh_params = [
        {'type': 'gradient', 'channel': 's', 'direct': 'x', 'thresh': (20, 100)},
        {'type': 'color', 'channel': 's', 'direct': None, 'thresh': (100, 255)},
        {'type': 'color', 'channel': 'gray', 'direct': None, 'thresh': (190, 255)}
    ]

    img = plt.imread(test_image)
    th = Threshold(img)
    for param in thresh_params:
        th.transform(param['channel'], param['direct'], thresh=param['thresh'])

        # Visualize the result in each step
        title1 = param['channel']
        if param['direct'] is None:
            title2 = 'color thresh ' + str(param['thresh'])
        else:
            title2 = param['direct'] + ' gradient thresh ' + str(param['thresh'])

        two_plots(th.channel_, th.binary, titles=(title1, title2, ''))

    threshed = th.binary
    plt.imshow(threshed, cmap='gray')
    plt.show()