"""
This file holds two classes for feature extraction:

- Class HogExtractor()
    HOG feature extractor class

- Class LbpExtractor()
    LBP feature extractor class
"""
import numpy as np
import cv2

from skimage.feature import hog as skimage_hog
from skimage.feature import local_binary_pattern as skimage_lbp

from utilities import change_colorspace


class HogExtractor(object):
    """HOG feature extractor class"""
    def __init__(self, colorspace='RGB', orient=9, pix_per_cell=(8, 8),
                 cell_per_block=(1, 1), block_normalization='L2',
                 visual=False):
        """Initialization

        :param colorspace: string
            Color space to use. Valid options are
            'GRAY', 'RGB' (default), 'HSV', 'HLS', 'YUV', 'YCrCb'.
        :param orient: int
            Number of orientations in HOG.
        :param pix_per_cell: tuple, 2x1
            Number of pixels per cell.
        :param cell_per_block: tuple, 2x1
            Number of cells per block.
        :param block_normalization: string
            Block normalization method
            ‘L1’, ‘L1-sqrt’, ‘L2’(default), ‘L2-Hys’
        :param visual: Bool
            True for visualization.
        """
        self._colorspace = colorspace
        self._orient = orient
        self._pix_per_cell = pix_per_cell
        self._cell_per_block = cell_per_block
        self._block_normalization = block_normalization

        self._visual = visual

    def extract(self, img, ravel=True):
        """Extract HOG features from a single image.

        :param img: numpy.ndarray
            Image array.
        :param ravel: Bool
            True for returning ravelled HOG features.

        :returns hog_features:
            If ravel == False, a list of numpy.ndarray
                Non-ravelled HOG features in different channels.
            If ravel == True, 1D numpy.array
                Ravelled HOG features.
        :returns hog_imgs: a list of 2D numpy.ndarray (if self.visual is True).
            HOG images in different channels.
        """
        hog_features = []
        hog_imgs = []

        # Change the color space
        img_new = change_colorspace(img, self._colorspace)

        if len(img_new.shape) == 2:
            # For gray scale image
            hog_features_single_channel, hog_img = self._hog(img_new)
            hog_features.append(hog_features_single_channel)
            hog_imgs.append(hog_img)
        else:
            # For color image
            for i in range(3):
                hog_features_single_channel, hog_img = \
                    self._hog(img_new[:, :, i])
                hog_features.append(hog_features_single_channel)
                hog_imgs.append(hog_img)

        if ravel is True:
            hog_features = np.concatenate(hog_features).ravel()

        if self._visual is True:
            return hog_features, hog_imgs
        else:
            return hog_features

    def sliding_window_extract(self, img, window_size=(64, 64),
                               step_size=(1.0, 1.0), scale=1.0):
        """Apply sliding window HOG feature extraction

        :param img: numpy.ndarray
            Image array.
        :param window_size: 1x2 tuple, int
            Sliding window size in (x, y).
        :param step_size: 1x2 tuple, float
            Sliding step in (x, y) in the unit of the image shape.
        :param scale: float
            Scale of the original image.

        :return: window_features: a list of 1D numpy.array
            Ravelled window features.
        :return: window_coordinates: a list of 2x2 tuple, int
            Window coordinates in ((x0, y0), (x1, y1))
        """
        img_resized = cv2.resize(
            img, (np.int(img.shape[1]*scale), np.int(img.shape[0]*scale)))

        # extract the features of the whole image
        hog_features = self.extract(img_resized, ravel=False)

        window_features = []
        window_coordinates = []
        y0_window = 0
        y1_window = window_size[1]
        x_step = int(step_size[0]*window_size[0])
        y_step = int(step_size[1]*window_size[1])
        # Apply sliding window
        while y1_window < img_resized.shape[0]:
            x0_window = 0
            x1_window = window_size[0]
            while x1_window < img_resized.shape[1]:
                # concatenate features in different channels
                combined = []
                for channel in hog_features:
                    # this is the tricky part
                    # For cell_per_block = 1, the feature shape is 8x8x1x1x9
                    # For cell_per_block = 2, the feature shape is 7x7x2x2x9
                    # For cell_per_block = 3, the feature shape is 6x6x3x3x9
                    x0_hog = x0_window // self._pix_per_cell[0]
                    x1_hog = x1_window // self._pix_per_cell[0] - self._cell_per_block[0] + 1
                    y0_hog = y0_window // self._pix_per_cell[1]
                    y1_hog = y1_window // self._pix_per_cell[1] - self._cell_per_block[1] + 1

                    combined.append(channel[y0_hog:y1_hog, x0_hog:x1_hog])

                window_features.append(np.concatenate(combined).ravel())
                window_coordinates.append(
                    ((np.int(x0_window/scale), np.int(y0_window/scale)),
                     (np.int(x1_window/scale), np.int(y1_window/scale))))

                x0_window += x_step
                x1_window += x_step

            y0_window += y_step
            y1_window += y_step

        return window_features, window_coordinates

    def _hog(self, img):
        """Extract the HOG features from a gray-scale image.

        :param img: 2D numpy.ndarray
            Image array.

        :returns hog_features: numpy.ndarray
            Un-ravelled HOG features.
        :returns hog_image:
            HOG image if self.visual is True.
        """
        if self._visual is True:
            # The run time when visualise=True is much longer!!!
            hog_features, hog_image = skimage_hog(
                img, orientations=self._orient,
                pixels_per_cell=self._pix_per_cell,
                cells_per_block=self._cell_per_block,
                transform_sqrt=False,
                block_norm=self._block_normalization,
                visualise=True,
                feature_vector=False)
        else:
            hog_image = None
            hog_features = skimage_hog(
                img, orientations=self._orient,
                pixels_per_cell=self._pix_per_cell,
                cells_per_block=self._cell_per_block,
                transform_sqrt=False,
                block_norm=self._block_normalization,
                visualise=False,
                feature_vector=False)

        return hog_features, hog_image


class LbpExtractor(object):
    """LBP feature extractor class"""
    def __init__(self, colorspace='GRAY', n_neighbors=8, radius=2, visual=False):
        """Initialization

        :param colorspace: string
            Color space to use. Valid options are
            'GRAY', 'RGB' (default), 'HSV', 'HLS', 'YUV', 'YCrCb'.
        :param n_neighbors: int
            Number of circularly symmetric neighbour set points
            (quantization of the angular space).
        :param radius: float
            Radius of circle (spatial resolution of the operator).
        :param visual: Bool
            True for visualization.
        """
        self._colorspace = colorspace
        self._n_neighbors = n_neighbors
        self._radius = radius

        self._visual = visual

    def extract(self, img):
        """Extract LBP features from an image

        :param img: numpy.ndarray
            Image array.

        :return:
            If self._visual is True:
                Return LBP images;
            If self._visual is False:
                Return flattened features.
        """
        # Change the color space
        img_new = change_colorspace(img, self._colorspace)

        lbp_imgs = []
        if len(img_new.shape) == 2:
            # For gray scale image
            lbp_img = self._lbp(img_new)
            lbp_imgs.append(lbp_img)
        else:
            # For color image
            for i in range(3):
                lbp_img = self._lbp(img_new[:, :, i])
                lbp_imgs.append(lbp_img)

        if self._visual is True:
            return lbp_imgs
        else:
            lbp_features = []
            for lbp_img in lbp_imgs:
                lbp_features.append(
                    np.histogram(lbp_img, bins=2 ** self._n_neighbors,
                                 range=(0, 2 ** self._n_neighbors))[0])

            return np.concatenate(lbp_features).astype(np.float32)

    def sliding_window_extract(self, img, window_size=(64, 64),
                               step_size=(1.0, 1.0), scale=1.0):
        """Apply sliding window LBP feature extraction

        :param img: numpy.ndarray
            Image array.
        :param window_size: 1x2 tuple, int
            Sliding window size in (x, y).
        :param step_size: 1x2 tuple, float
            Sliding step in (x, y) in the unit of the image shape.
        :param scale: float
            Scale of the original image.

        :return: window_features: a list of 1D numpy.array
            Ravelled window features.
        :return: window_coordinates: a list of 2x2 tuple, int
            Window coordinates in ((x0, y0), (x1, y1))
        """
        img_resized = cv2.resize(
            img, (np.int(img.shape[1]*scale), np.int(img.shape[0]*scale)))

        window_features = []
        window_coordinates = []
        x_step = int(step_size[0]*window_size[0])
        y_step = int(step_size[1]*window_size[1])
        y0_window = 0
        y1_window = window_size[1]
        # Apply sliding window
        while y1_window < img_resized.shape[0]:
            x0_window = 0
            x1_window = window_size[0]
            while x1_window < img_resized.shape[1]:
                window_img = img_resized[y0_window:y1_window, x0_window:x1_window]
                window_features.append(self.extract(window_img))
                window_coordinates.append(
                    ((np.int(x0_window/scale), np.int(y0_window/scale)),
                     (np.int(x1_window/scale), np.int(y1_window/scale))))

                x0_window += x_step
                x1_window += x_step

            y0_window += y_step
            y1_window += y_step

        return window_features, window_coordinates

    def _lbp(self, img):
        """Extract the LBP features from an image

        :param img: 2D numpy.ndarray
            Image array.

        :return lbp_img: 2D numpy.ndarray
            LBP image array.
        """
        lbp_img = skimage_lbp(img, self._n_neighbors, self._radius)

        return lbp_img
