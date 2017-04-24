#!/usr/bin/python

"""
CarClassifier class
  
@method _change_colorspace():
    Convert the color space of an image.
    Called by extract() method.

@method _load():
    Read features from image files.

@method extract():
    Extract features from an image.
    Called by train() method.

@method train():
    Train a car classifier.

@method predict():
    Predict an image data (set).

"""

import glob
import pickle
import time

import numpy as np
import scipy
import matplotlib.pyplot as plt
import cv2

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

from skimage.feature import hog as skimage_hog
from skimage.feature import local_binary_pattern as skimage_lbp


DEBUG = False


class CarClassifier(object):
    """A car classifier object"""
    def __init__(self, shape=(64, 64), color_space='BGR', classifier=None,
                 lbp=True, hog=True, channel=0,
                 hog_orient=9, hog_pix_per_cell=(8, 8), hog_cell_per_block=(1, 1),
                 lbp_n_neighbors=8, lbp_radius=4, lbp_n_bins=256):
        """Initialization

        Parameters
        ----------
        shape: tuple, 2x1
            Image shape in the training/testing data set.
        color_space: string
            Color space to use. Valid options are
            'BGR' (default), 'HSV', 'HLS', 'YUV', 'YCrCb'.
        classifier: object
            A classifier instance, e.g. LinearSVC, DecisionTreeClassifier
        hog: Boolean
            True for including HOG features.
        lbp: Boolean
            True for including LBP features.
        channel: 0, 1, 2 or 'all'
            Channel(s) for features extraction.
        hog_orient: int
            Number of orientations in HOG.
        hog_pix_per_cell: tuple, 2x1
            Number of pixels per cell.
        hog_cell_per_block: tuple, 2x1
            NUmber of celss per block.
        lbp_n_neighbors: int
            Number of circularly symmetric neighbour set points
            (quantization of the angular space).
        lbp_radius: float
            Radius of circle (spatial resolution of the operator).
        lbp_n_bins: int
            Number of bins in the histogram of LBP features.
        """
        if classifier is None:
            self.cls = LinearSVC()
        else:
            self.cls = classifier

        # scaler is an attribute since it will be used by both
        # the train() and predict() methods.
        self.scaler = StandardScaler()

        # self.selector = SelectKBest(k='all')
        # self.selector = PCA(n_components=100)

        self.shape = shape

        self.color_space = color_space
        self._channels = None
        self.channels = channel

        self.hog = hog
        self.lbp = lbp
        if self.hog is False and self.lbp is False:
            raise ValueError("At least one algorithm is required!")

        self.hog_orient = hog_orient
        self.hog_pix_per_cell = hog_pix_per_cell
        self.hog_cell_per_block = hog_cell_per_block

        self.lbp_n_neighbors = lbp_n_neighbors
        self.lbp_radius = lbp_radius
        self.lbp_n_bins = lbp_n_bins

        self.feature_shape = None

    @property
    def channels(self):
        """The 'channels' property"""
        return self._channels

    @channels.setter
    def channels(self, value):
        """"""
        if value == 'all':
            self._channels = range(3)
        elif value in [0, 1, 2]:
            self._channels = [value]
        else:
            raise ValueError("Unknown channel!")

    def _change_colorspace(self, img):
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
        if self.color_space not in ('BGR', 'RGB'):
            if self.color_space == 'HSV':
                new_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif self.color_space == 'LUV':
                new_img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
            elif self.color_space == 'HLS':
                new_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            elif self.color_space == 'YUV':
                new_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            elif self.color_space == 'YCrCb':
                new_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            else:
                raise ValueError("Unknown color space!")
        else:
            new_img = np.copy(img)

        return new_img

    def extract(self, img, window_shape=None, slide_step=None,
                x_range=None, y_range=None, x_pad=0.0):
        """
        Extract features from an image.

        Parameters
        ----------
        img: numpy.ndarray
            Image array.
        window_shape: tuple, 2x1
            Shape of the window for feature extraction, in pixel.
        slide_step: tuple, 2x1
            Advance steps of the sliding window in both dimensions,
            in pixel.
        x_range: tuple, 2x1
            x range of the window to search, in pixel.
        y_range: tuple, 2x1
            y range of the window to search, in pixel.
        x_pad: int
            Width of the pad at each side of the image. The padding
            areas are of importance to have more counts (heat) when
            the car is on the edge of the image.

        Returns
        -------
        features: 2D numpy.ndarray, [index, features]
            Features of the sliding windows.
        windows: list of ((x0, y0), (x1, y1))
            Diagonal coordinates of the sliding windows.

        The features must be extracted from a window with the size
        of "shape" attribute.
        """
        assert img.dtype == np.uint8

        if window_shape is None:
            window_shape = self.shape
        if slide_step is None:
            slide_step = window_shape
        if x_range is None:
            x_range = (0, window_shape[1])
        if y_range is None:
            y_range = (0, window_shape[0])

        # Window of the processing area
        y0 = np.int(y_range[0])
        y1 = np.int(y_range[1])
        x0 = np.int(x_range[0])
        x1 = np.int(x_range[1])

        if DEBUG is True:
            print("Shape of the original image: {}".format(img.shape))

        t0 = time.time()

        # Augment the feature_img by 2*x_pad
        feature_img = np.zeros((y1 - y0, x1 - x0 + np.int(2*x_pad), 3),
                               dtype=np.uint8)
        feature_img[:, np.int(x_pad):x1 - x0 + np.int(x_pad)] \
            = self._change_colorspace(img[y0:y1, x0:x1])

        if DEBUG is True:
            feature_img_original = np.zeros_like(feature_img)
            feature_img_original[:, np.int(x_pad):x1 - x0 + np.int(x_pad), :] \
                = img[y0:y1, x0:x1, :]
            print("Shape of the augmented feature image: {}"
                  .format(feature_img.shape))
            print("Time for color space conversion: {:.4} s"
                  .format(time.time() - t0))

        # Scaling factor
        scale = np.array([window_shape[1]/self.shape[1],
                          window_shape[0]/self.shape[0]])

        if DEBUG is True:
            print("Scaling factor: {}".format(scale))

        if scale[0] != 1.0 or scale[1] != 1.0:
            # Scale the feature image.
            img_shape = feature_img.shape[0:2]
            feature_img_scaled = cv2.resize(
                feature_img, (np.int(img_shape[1]/scale[1]),
                              np.int(img_shape[0]/scale[0])))
        else:
            feature_img_scaled = np.copy(feature_img)

        #
        # Extract features from the whole image first.
        # This is much faster than extract features from the individual
        # Sub-images.
        #

        # Extract the HOG features.
        hog_features_by_channel = []
        hog_images = []
        hog_times = []
        if self.hog is True:
            t0 = time.time()
            for i in self.channels:
                if DEBUG is True:
                    # The run time when visualise=True is much longer!!!
                    hog_features, hog_image = skimage_hog(
                        feature_img_scaled[:, :, i], orientations=self.hog_orient,
                        pixels_per_cell=self.hog_pix_per_cell,
                        cells_per_block=self.hog_cell_per_block,
                        transform_sqrt=False,
                        visualise=True,
                        feature_vector=False)
                    hog_images.append(hog_image)
                else:
                    hog_features = skimage_hog(
                        feature_img_scaled[:, :, i], orientations=self.hog_orient,
                        pixels_per_cell=self.hog_pix_per_cell,
                        cells_per_block=self.hog_cell_per_block,
                        transform_sqrt=False,
                        visualise=False,
                        feature_vector=False)

                hog_features_by_channel.append(hog_features)

            hog_times.append(time.time() - t0)

        # Extract the LBP features.
        lbp_img_by_channel = []
        lbp_img = None
        lbp_times = []
        if self.lbp is True:
            t0 = time.time()
            for i in self.channels:
                lbp_img = skimage_lbp(feature_img_scaled[:, :, i],
                                      self.lbp_n_neighbors, self.lbp_radius)

                lbp_img_by_channel.append(lbp_img)

            lbp_times.append(time.time() - t0)

        features = []  # A list of combined features
        windows = []  # A list of the diagonal coordinates in original image

        dt_lbphist = 0.0

        # The following code is dealing with the re-sized image, so
        # self.shape should be used!
        # x0(y0)_window_origin are used to minic the window sliding on
        # the original image. Therefore, shape should be used.
        y0_window = 0
        y0_window_origin = 0
        while (y0_window + self.shape[0]) <= feature_img_scaled.shape[0]:
            x0_window = 0
            x0_window_origin = 0
            while (x0_window + self.shape[1]) <= feature_img_scaled.shape[1]:
                window_features = []
                hog_features = []
                lbp_features = []

                # Extract HOG features of the sliding window
                hog_y0 = y0_window // self.hog_pix_per_cell[1]
                hog_y1 = (y0_window + self.shape[0]) // self.hog_pix_per_cell[1] - 1
                hog_x0 = x0_window // self.hog_pix_per_cell[0]
                hog_x1 = (x0_window + self.shape[1]) // self.hog_pix_per_cell[0] - 1

                for ch in hog_features_by_channel:
                    hog_features.append(ch[hog_y0:hog_y1, hog_x0:hog_x1].ravel())

                hog_features = np.concatenate(hog_features, axis=0)
                window_features.append(hog_features)

                # Extract LBP features of the sliding window
                if self.lbp is True:
                    t0_lbphist = time.time()
                    for ch in lbp_img_by_channel:
                        lbp_features.append(
                            np.histogram(ch[y0_window:(y0_window + self.shape[0]),
                                         x0_window:(x0_window + self.shape[1])],
                                         bins=self.lbp_n_bins,
                                         range=(0, 2**self.lbp_n_neighbors))[0])

                    lbp_features = np.concatenate(lbp_features, axis=0)
                    window_features.append(lbp_features)

                    if DEBUG is True:
                        dt_lbphist += time.time() - t0_lbphist

                sub_img = feature_img_scaled[
                          y0_window:(y0_window + self.shape[0]),
                          x0_window:(x0_window + self.shape[1])]

                if DEBUG:
                    print("Shape of the sub-image: {}".format(sub_img.shape))

                window_features = np.concatenate(window_features)

                features.append(window_features)

                windows.append(((x0 + x0_window_origin - x_pad,
                                 y0 + y0_window_origin),
                                (x0 + x0_window_origin + window_shape[1] - x_pad,
                                 y0 + y0_window_origin + window_shape[0])))

                if DEBUG is True:
                    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
                    print(feature_img_original.shape)
                    ax[0].imshow(feature_img_original[
                        y0_window_origin:y0_window_origin + window_shape[0],
                        x0_window_origin:x0_window_origin + window_shape[1]])
                    ax[0].set_title('Original image', fontsize=20)

                    ax[1].imshow(hog_image[y0_window: y0_window + self.shape[0],
                                 x0_window: x0_window + self.shape[1]])
                    ax[1].set_title('HOG image', fontsize=20)
                    ax[2].plot(hog_features)
                    ax[2].set_title('hog features', fontsize=20)

                    if self.lbp is True:
                        ax[2, 0].imshow(lbp_img)
                        ax[2, 0].set_title('LBP image', fontsize=20)
                        ax[2, 1].plot(lbp_features)
                        ax[2, 1].set_title('LBP features', fontsize=20)

                    plt.tight_layout()
                    plt.show()

                x0_window += np.int(slide_step[1]/scale[1])
                x0_window_origin += slide_step[1]
            y0_window += np.int(slide_step[0]/scale[0])
            y0_window_origin += slide_step[0]

        if DEBUG is True:
            if self.lbp is True:
                print("Time for extracting LBP-histogram features: {:.4} s"
                      .format(dt_lbphist))

        return np.array(features, dtype=np.float32), windows

    def _load(self, files, max_files):
        """Read features from image files.

        Parameters
        ----------
        files: list
            List of file names.
        max_files:
            Maximum number of files to read.

        Returns
        -------
        Features of images in numpy.ndarray.
        """
        features = []
        for file in files:
            img = cv2.imread(file)
            assert img.dtype == np.uint8

            if img.shape != self.shape:
                img = cv2.resize(img, self.shape)

            # window_features, and windows are both lists
            window_features, windows = self.extract(img)

            if self.feature_shape is None:
                self.feature_shape = window_features[0].shape
            else:
                assert self.feature_shape == window_features[0].shape

            features.append(window_features[0])
            if isinstance(max_files, int) and len(features) > max_files:
                break

        return np.array(features, dtype=np.float32)

    def train(self, cars, non_cars, max_images=None, test_size=0.3,
              random_state=None):
        """Train a car classifier.

        Parameters
        ----------
        cars: list
            List of car file names.
        non_cars: list
            List of non-car file names.
        max_images: int
            Maximum number of file to read in each data set.
        test_size: float, in (0, 1)
            Percent of test data in self.X.
        random_state: int or None.
            Pseudo-random number generator state used for random sampling.
        """
        car_features = self._load(cars, max_images)
        noncar_features = self._load(non_cars, max_images)

        X = np.vstack((car_features, noncar_features))
        y = np.hstack((np.ones(len(car_features)),
                       np.zeros(len(noncar_features)))).astype(np.int8)

        X_shuffle, y_shuffle = shuffle(X, y)

        # Cross validation to determine the best hyperparameters
        # scores = cross_val_score(
        #     self.cls, self.scaler.fit_transform(X_shuffle), y_shuffle, cv=10)
        # print("Mean prediction scores: {:.4f}".format(scores.mean()))

        X_train, X_test, y_train, y_test = train_test_split(
            X_shuffle, y_shuffle, test_size=test_size, random_state=random_state)

        print("Number of training data: {}".format(len(y_train)))
        print("Number of testing data: {}".format(len(y_test)))
        print("Number of features: {}".format(len(X_train[0])))

        t0 = time.time()

        normalized_X = self.scaler.fit_transform(X_train)
        self.cls.fit(normalized_X, y_train)
        print("Training finished in {:.1f} s".format(time.time() - t0))

        y_pred = self.predict(X_test)
        print("Prediction accuracy on test set: {}".
              format(float(sum(y_pred == y_test) / len(y_pred))))

    def predict(self, X):
        """Predict an image data (set).

        Parameters
        ----------
        X: numpy.ndarray
            Features.

        Returns
        -------
        y_pred: numpy.ndarray
            Predicted labels.
        """
        return self.cls.predict(self.scaler.transform(X)).astype(np.int8)


if __name__ == "__main__":
    car_files = glob.glob("data/vehicles/KITTI_extracted/*.png")
    car_files.extend(glob.glob("data/vehicles/GTI_Far/*.png"))
    car_files.extend(glob.glob("data/vehicles/GTI_Left/*.png"))
    car_files.extend(glob.glob("data/vehicles/GTI_Right/*.png"))
    car_files.extend(glob.glob("data/vehicles/GTI_MiddleClose/*.png"))
    noncar_files = glob.glob("data/non-vehicles/Extras/*.png")
    noncar_files.extend(glob.glob("data/non-vehicles/GTI/*.png"))

    cls = LinearSVC(C=0.0001)
    # cls = DecisionTreeClassifier(max_depth=10)
    # cls = RandomForestClassifier(n_estimators=20, max_depth=6)
    car_cls = CarClassifier(
        classifier=cls, color_space='YCrCb', channel='all', lbp=False)

    car_cls.train(car_files, noncar_files, max_images=1000)

    output = 'car_classifier.pkl'
    with open(output, "wb") as fp:
        pickle.dump(car_cls, fp)
        print("Car classifier was saved in {}".format(output))

    # Apply the classifier on a test image
    test_image = 'test_images/test_image_white_car.png'
    test_img = cv2.imread(test_image)

    features, windows = car_cls.extract(
        test_img, window_shape=(64, 64), slide_step=(8, 8),
        x_range=(300, 1280), y_range=(200, 296), x_pad=64)

    predictions = car_cls.predict(features)

    print(predictions)

    car_cls.extract(test_img)

