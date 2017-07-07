#!/usr/bin/python
"""
Class:
    - LaneLine()

Functions:
    - find_peaks()
    - sample_lines()

"""
import numpy as np


class LaneLine(object):
    """Lane line in the bird's eye view"""
    def __init__(self, y, order=2, max_fail=25):
        """Initialization

        :param order: int
            Order of the polynomial fit.
        """
        # Consecutive frames in which a good fit is not found.
        # If i_fail > max_fail, a blind search will be launched.
        # Otherwise, update_on_image method will be called.
        self._max_fail = max_fail
        self._i_fail = max_fail

        self.order = order
        self._points = None  # detected points
        self.points = None  # points property
        self.p_fit = None  # polynomial coefficients for the current fit
        self.y_fit = y
        self.x_fit = None  # x location (in pixel)
        self.pfit_good = None  # polynomial coefficients for the last good fit

        self.local_bend_radius = []
        self.ahead_bend_angle = []

    @property
    def points(self):
        """points getter"""
        return self._points

    @points.setter
    def points(self, value):
        """points setter"""
        if value is None or len(value) == 0:
            self.p_fit = None
            self.x_fit = None
            self._i_fail += 1
        else:
            self._points = value
            self._fit()

    def _fit(self):
        """Polynomial fit"""
        self.p_fit = np.polyfit(self.points[1], self.points[0], self.order)
        self.x_fit = np.poly1d(self.p_fit)(self.y_fit)
        self._i_fail = 0


def find_peaks(x, width=300, step=50, min_intensity=100,
               background_intensity=20):
    """Find peaks in a 1D array

    :param x: 1d array like
        Input 1d data.
    :param width: int
        Width of the moving window
    :param step: int
        Step size of the moving window search.
    :param min_intensity: int
        Minimum intensity of a peak.
    :param background_intensity: int
        Intensity below this value is considered as background.

    :return peaks: list
        A list of peak index.
    """
    peaks = []
    i = 0
    while i + width < len(x):
        # We expect a sharp peak. Therefore, the intensities of most points
        # should be not bigger than the background intensity
        if sum(x[i:i + width] <= background_intensity) < width/2:
            pass
        else:
            peak = np.argmax(x[i:(i + width)])
            if peak == 0 or peak == width - 1:
                # skip fake peaks
                pass
            elif x[i + peak] > min_intensity:
                if i + peak not in peaks:
                    peaks.append(i + peak)

        i += step

    return peaks


def sample_lines(img, width=200, step=20):
    """Sample the possible points in lines in a binary image

    :param img: numpy.ndarray
        Original image.
    :param width: int
        Window width for a single line.
    :param step: int
        Step size when performing sliding window search vertically.

    :return points: list of [[x], [y]], x and y are both lists
        Points in lines.
    """
    y1 = img.shape[0]
    y0 = y1 - int(img.shape[0]/5)
    window = img[y0:y1, :]

    convolved = np.convolve(
        np.sum(window, axis=0), np.ones(10), mode='same')
    peaks = find_peaks(convolved)
    points = []
    if len(peaks) == 0:
        return points
    else:
        conv_window = np.ones(10)
        for peak0 in peaks:
            x0 = max(0, peak0 - int(width/2))
            x1 = min(img.shape[1], peak0 + int(width/2))
            y1 = img.shape[0]
            y0 = y1 - step
            peak = peak0
            x = []
            y = []
            while y0 >= 0:
                x_sum = np.sum(img[y0:y1, x0:x1], axis=0)
                convolved = np.convolve(x_sum, conv_window, mode='same')

                if max(convolved) > 0:
                    peak = int((np.where(convolved > 0.95 * max(convolved))[0]).mean())
                    peak += x0
                    x.append(peak)
                    y.append(int(y1 - step/2))

                y1 -= step
                y0 -= step

                x0 = max(0, peak - int(width / 2))
                x1 = min(img.shape[1], peak + int(width / 2))

            points.append((x, y))

        return points
