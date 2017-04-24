#!/usr/bin/python
"""
This file holds three classes:

CurveLine:
    Parent class of TwinLine and LaneLine.

LaneLine:
    single-line.

TwinLine:
    Double-line.

"""
from abc import abstractmethod
from time import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


DEBUG_LINE = False

INF = 1.0e21
Y_METER_PER_PIXEL = 30.0 / 720  # meters per pixel in y dimension
X_METER_PER_PIXEL = 3.7 / 900  # meters per pixel in x dimension


class CurveLine(object):
    """CurveLine class"""
    def __init__(self, y, order=2):
        """Instantiate the CurveLine class

        Parameters
        ----------
        y: 1-D array-like
            Vertical coordinates of the line.
            use in fitting x=f(y)
        order: int
            Order of the polynomial fit.
        """
        # polynomial coefficients for the current fit
        self.order = order
        self.p = np.array([])

        self.y = np.asarray(y)
        self.x = np.array([])

        self.p_hst = []
        self.x_hst = []

        self.local_bend_radius = None
        self.ahead_bend_angle = None

    @abstractmethod
    def search(self, img):
        """Search points which belong to lines in an image"""
        raise NotImplementedError


class TwinLine(CurveLine):
    """TwinLine class"""
    def __init__(self, y, div=0.5):
        """Initialize the TwinLine class

        Parameters
        ----------
        y: 1D array like
            Vertical coordinates of the line.
            use in fitting x=f(y)
        div: float, 0.0 < div < 1.0
            The location to divide the left and right part of an image.
        """
        super().__init__(y)

        # Left and right LaneLine objects
        self.left = LaneLine(y)
        self.right = LaneLine(y)
        self.div = div
        self.is_parallel = False

        # distance in meters of vehicle center from the line
        self.left_space = None
        self.right_space = None

    def search(self, img):
        """Search lines in an image

        Parameter
        ---------
        img: numpy.ndarray()
            Image array.
        """
        width = img.shape[1]

        self.left.search(img, x_lim=(0.0, self.div))
        self.right.search(img, x_lim=(self.div, 1.0))

        if DEBUG_LINE:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.imshow(img, cmap='gray')
            ax.scatter(self.left.x_detected, self.left.y_detected, color='green', marker='*', s=200)
            ax.scatter(self.right.x_detected, self.right.y_detected, color='blue', marker='*', s=200)
            if self.left.x.any():
                ax.plot(self.left.x, self.left.y, '--', color='purple', lw=10, alpha=0.7)
            if self.left.y.any():
                ax.plot(self.right.x, self.right.y, '--', color='purple', lw=10, alpha=0.7)

            plt.tight_layout()
            ### plt.savefig('output_images/line_fit.png')
            plt.show()

        # The local bend radius and ahead bend angle are calculated by
        # the average value if both lines have the average poly-fit.
        # Otherwise, they will only follow the only line that has the
        # average poly-fit.
        if self.left.ave_p.any() and self.right.ave_p.any():
            self.local_bend_radius = \
                (self.left.local_bend_radius + self.right.local_bend_radius) / 2.0
            self.ahead_bend_angle = \
                (self.left.ahead_bend_angle + self.right.ahead_bend_angle) / 2.0
            self.left_space = (width * self.div - self.left.ave_x[-1]) * X_METER_PER_PIXEL
            self.right_space = (self.right.ave_x[-1] - width * self.div) * X_METER_PER_PIXEL
        elif self.left.ave_p.any():
            self.local_bend_radius = self.left.local_bend_radius
            self.ahead_bend_angle = self.left.ahead_bend_angle
            self.left_space = (width * self.div - self.left.ave_x[-1]) * X_METER_PER_PIXEL
            self.right_space = None
        elif self.right.ave_p.any():
            self.local_bend_radius = self.right.local_bend_radius
            self.ahead_bend_angle = self.right.ahead_bend_angle
            self.left_space = None
            self.right_space = (self.right.ave_x[-1] - width * self.div) * X_METER_PER_PIXEL


class LaneLine(CurveLine):
    """Single lane line class"""
    def __init__(self, y, max_fail=25):
        """Initialization

        Parameters
        ----------
        y: 1D array like
            Vertical coordinates of the line.
            use in fitting x=f(y).
        max_fail: int
            Maximum allowed poor fit before re-search the whole area.
        """
        super().__init__(y)

        # Coordinates of the detected line pixels
        self.x_detected = []
        self.y_detected = []

        # Consecutive frames in which a good fit is not found.
        # If i_fail > max_fail, a blind search will be launched.
        # Otherwise, update_on_image method will be called.
        self.max_fail = max_fail
        self.i_fail = max_fail

        # Were x, y detected?
        self._detected = False
        self.detected = False

        # Is current fit good?
        self._goodness = False
        self.goodness = False

        # Average history value
        self.ave_x = np.array([])
        self.ave_p = np.array([])

    @property
    def goodness(self):
        return self._goodness

    @goodness.setter
    def goodness(self, value):
        if not isinstance(value, bool):
            raise ValueError("value must be a Boolean.")
        self._goodness = value
        if not value:
            self.i_fail += 1
            if self.i_fail > self.max_fail:
                self.ave_p = np.array([])
                self.ave_x = np.array([])
                self.ahead_bend_angle = None
                self.local_bend_radius = None
        else:
            self.i_fail = 0

    @property
    def detected(self):
        return self._detected

    @detected.setter
    def detected(self, value):
        if not isinstance(value, bool):
            raise ValueError("value must be a Boolean.")

        self._detected = value
        if not value:
            self.goodness = False

    def fit(self):
        """Fit x=f(y)."""

        def func(p, y, x):
            return p[0] * y ** 2 + p[1] * y + p[2] - x

        if len(self.x_detected) == 0:
            self.detected = False
        else:
            self.detected = True

            # t0 = time()
            if self.ave_p.any() and self.i_fail <= self.max_fail:
                # Use np.polyfit for speed when searching around the
                # trusted region.
                self.p = np.polyfit(self.y_detected, self.x_detected, self.order)
            else:
                p0 = np.array([0, 1, 0])
                res_slq = least_squares(func, p0, loss='soft_l1', f_scale=1.0,
                                        ftol=1e-8, xtol=1e-8,
                                        args=(self.y_detected, self.x_detected))
                self.p = res_slq.x

            # print("{:.1f} ms".format(1000*(time() - t0)))

            self.x = np.poly1d(self.p)(self.y)

            if self.i_fail <= self.max_fail:
                if self.ave_p.any():
                    flag = self._check_consistency()
                    i = 0
                    length = len(self.y_detected)
                    sample_size = int(length * 0.8)
                    if sample_size >= 5:
                        while flag is False and i < 20:
                            i += 1

                            sample_index = np.random.choice(
                                np.arange(length), sample_size)

                            self.p = np.polyfit(self.y_detected[sample_index],
                                                self.x_detected[sample_index],
                                                self.order)
                            self.x = np.poly1d(self.p)(self.y)

                            flag = self._check_consistency()

                    if flag is True:
                        self.goodness = True
                        w = 0.2 + 0.3*float(self.i_fail/self.max_fail)
                        self.ave_p = (1 - w) * self.ave_p + w*self.p
                    else:
                        self.goodness = False
                else:
                    raise ValueError('Not expected in this region!')
            else:
                self.goodness = self._check_goodness()
                if self.goodness is True:
                    self.ave_p = self.p

            if self.ave_p.any():
                self.ave_x = np.poly1d(self.ave_p)(self.y)
                self.local_bend_radius = self._bend_radius(self.ave_p, self.y)
                self.ahead_bend_angle = self._bend_angle(self.ave_x, self.y)

    @staticmethod
    def _bend_radius(p, y):
        """"""
        A = p[0] / Y_METER_PER_PIXEL ** 2 * X_METER_PER_PIXEL
        B = p[1] / Y_METER_PER_PIXEL * X_METER_PER_PIXEL

        return (1 + (2*A*y[-1] + B)**2)**1.5/(2*A)

    @staticmethod
    def _bend_angle(x, y):
        """"""
        bend_angle = np.arctan((x[0] - x[2])/(y[2] - y[0])
                               *X_METER_PER_PIXEL/Y_METER_PER_PIXEL)

        return bend_angle*180/np.pi

    def _check_consistency(self):
        """Check the consistency of the current fit"""
        dx = self.x - self.ave_x
        tol = 25 + 15*float(self.i_fail/self.max_fail)
        if dx.std() > tol:
            return False
        else:
            return True

    def _check_goodness(self):
        """Check the goodness of the current line"""
        # Too few points detected
        if abs(self._bend_radius(self.p, self.y)) < 100:
            return False

        return True

    def search(self, img, x_lim=(0.0, 1.0), p0=None, conv_window=10,
               jitter=100, moving_window=50, moving_step=25,
               min_peak_intensity=100):
        """Search the possible points in a line

        Parameters
        ----------
        img: numpy.ndarray
            Image array
        x_lim: tuple like, (left, right)
            Fractional x boundary of the image to search.
        p0: tuple like
            Initial value of the polynomial coefficients.
        conv_window: int
            Width of the window for convolution.
        jitter: int
            Search range (in x) along the reference line.
        moving_window: int
            Size of the moving window.
        moving_step: int
            Step size of the window moving in y direction.
        min_peak_intensity: int
            Minimum intensity to be recognized as a peak.
        """
        width = img.shape[1]
        height = img.shape[0]

        # Search the whole image if the reference is not given
        if not self.ave_p.any():
            ref = np.vstack([np.ones(height)*int((x_lim[0] + x_lim[1])*width/2),
                             np.arange(height)]).T
            jitter = int((x_lim[1] - x_lim[0])*width/2)
        else:
            ref = np.vstack([self.ave_x, self.y]).T

        x = []
        y = []
        window = np.ones(conv_window)
        start = height - moving_window
        while start > 0:
            y_center = start + int(moving_window / 2)

            x0 = int(ref[y_center, 0] - jitter)
            if x0 < 0:
                x0 = 0
            x1 = int(ref[y_center, 0] + jitter)
            if x1 > img.shape[1]:
                x1 = img.shape[1]
            if x1 <= 0 or x0 >= img.shape[1]:
                start -= moving_step
                continue

            img_slice = img[start:start + moving_window, x0:x1]
            sum = np.sum(img_slice, axis=0)
            conv = np.convolve(window, sum, mode='same')

            if max(conv) > min_peak_intensity:
                peak = (np.where(conv > 0.95*max(conv))[0]).mean()
                x.append(peak + x0)
                y.append(y_center)

                ### For debug
                # fig, ax = plt.subplots(3, 1)
                # ax[0].imshow(img_slice)
                # ax[0].set_xlim([0, img_slice.shape[1]])
                # ax[1].plot(sum)
                # ax[1].set_xlim([0, img_slice.shape[1]])
                # ax[1].set_title('Sum', fontsize=12)
                # ax[2].plot(conv)
                # ax[2].scatter(peak, conv[int(peak)], c='red', s=100)
                # ax[2].set_xlim([0, img_slice.shape[1]])
                # ax[2].set_title('Conv', fontsize=12)
                # plt.tight_layout()
                # plt.show()

            start -= moving_step

        if len(x) == 0:
            self.detected = False
        else:
            self.x_detected = np.asarray(x)
            self.y_detected = np.asarray(y)
            self.fit()
