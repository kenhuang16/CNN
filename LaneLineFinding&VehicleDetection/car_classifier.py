#!/usr/bin/python

"""
CarClassifier class
  
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
    def __init__(self, shape=(64, 64), classifier=None, lbp=True, hog=True,
                 hog_orient=9, hog_pix_per_cell=(8, 8), hog_cell_per_block=(1, 1),
                 lbp_n_neighbors=8, lbp_radius=2):
        """Initialization

        Parameters
        ----------
        shape: tuple, 2x1
            Image shape in the training/testing data set.
        classifier: object
            A classifier instance, e.g. LinearSVC, DecisionTreeClassifier
        hog: Boolean
            True for including HOG features.
        lbp: Boolean
            True for including LBP features.
        hog_orient: int
            Number of orientations in HOG.
        hog_pix_per_cell: tuple, 2x1
            Number of pixels per cell.
        hog_cell_per_block: tuple, 2x1
            Number of cells per block.
        lbp_n_neighbors: int
            Number of circularly symmetric neighbour set points
            (quantization of the angular space).
        lbp_radius: float
            Radius of circle (spatial resolution of the operator).
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

        self.hog = hog
        self.lbp = lbp
        if self.hog is False and self.lbp is False:
            raise ValueError("At least one algorithm is required!")

        self._hog_orient = hog_orient
        self._hog_pix_per_cell = hog_pix_per_cell
        self._hog_cell_per_block = hog_cell_per_block

        self._lbp_n_neighbors = lbp_n_neighbors
        self._lbp_radius = lbp_radius

        self.feature_shape = None

    def extract(self, img, window_shape=None, sliding_step=None):
        """
        Extract features from an image.

        Parameters
        ----------
        img: numpy.ndarray
            Image array.
        window_shape: tuple, 2x1
            Shape of the window for feature extraction, in pixel. One
            image normally contains multiple windows during image scan.
        sliding_step: tuple, 2x1
            Advance steps of the sliding window in both dimensions,
            in pixel.

        Returns
        -------
        window_features: list of 2D numpy.ndarray
            Features of the sliding windows.
        windows: list of ((x0, y0), (x1, y1))
            Coordinates of the sliding windows.

        """
        assert img.dtype == np.uint8

        # LBP applies to a grayscale image
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = np.copy(img)

        # Set default values for window_shape and sliding_step
        if window_shape is None:
            window_shape = self.shape
        if sliding_step is None:
            sliding_step = window_shape

        if DEBUG is True:
            print("Shape of the original image: {}".format(img.shape))

        # Scaling factor
        # The image should be scaled before feature extraction in order
        # to make the window size the same as self.shape.
        scaling_factor = np.array([window_shape[1]/self.shape[1],
                                   window_shape[0]/self.shape[0]])

        if DEBUG is True:
            print("Scaling factor: {}".format(scaling_factor))

        if scaling_factor[0] != 1.0 or scaling_factor[1] != 1.0:
            # Scale the image.
            img_shape = img_gray.shape
            img_gray = cv2.resize(
                img_gray, (np.int(img_shape[1] / scaling_factor[1]),
                           np.int(img_shape[0] / scaling_factor[0])))
        else:
            img_gray = np.copy(img_gray)

        #
        # Extract features from the whole image first.
        # This is much faster than extract features from the individual
        # Sub-images.
        #

        # Extract the HOG features.
        hog_features = None
        hog_image = None
        hog_times = []
        if self.hog is True:
            t0 = time.time()

            if DEBUG is True:
                # The run time when visualise=True is much longer!!!
                hog_features, hog_image = skimage_hog(
                    img_gray, orientations=self._hog_orient,
                    pixels_per_cell=self._hog_pix_per_cell,
                    cells_per_block=self._hog_cell_per_block,
                    transform_sqrt=False,
                    visualise=True,
                    feature_vector=False)
            else:
                hog_features = skimage_hog(
                    img_gray, orientations=self._hog_orient,
                    pixels_per_cell=self._hog_pix_per_cell,
                    cells_per_block=self._hog_cell_per_block,
                    transform_sqrt=False,
                    visualise=False,
                    feature_vector=False)

            hog_times.append(time.time() - t0)

        # Extract the LBP features.
        # Histogram has not been applied here!
        lbp_img = None
        lbp_times = []
        if self.lbp is True:
            t0 = time.time()

            lbp_img = skimage_lbp(
                img_gray, self._lbp_n_neighbors, self._lbp_radius)

            lbp_times.append(time.time() - t0)

        #
        # In the following code, the sliding window is applied in both
        # the original image and the scaled image. The features of the
        # scaled image are stored while the coordinates of the windows
        # in the original image are stored.
        #
        window_features = []  # A list of window features
        windows_ = []  # A list of window coordinates in original image

        y0_window = 0
        y0_window_origin = 0
        while (y0_window + self.shape[0]) <= img_gray.shape[0]:
            x0_window = 0
            x0_window_origin = 0
            while (x0_window + self.shape[1]) <= img_gray.shape[1]:
                combined_features = []

                # Extract HOG features of the sliding window
                if self.hog is True:
                    hog_y0 = y0_window // self._hog_pix_per_cell[1]
                    hog_y1 = (y0_window + self.shape[0]) // self._hog_pix_per_cell[1] - 1
                    hog_x0 = x0_window // self._hog_pix_per_cell[0]
                    hog_x1 = (x0_window + self.shape[1]) // self._hog_pix_per_cell[0] - 1

                    combined_features.append(
                        hog_features[hog_y0:hog_y1, hog_x0:hog_x1].ravel())

                # Extract LBP features of the sliding window
                if self.lbp is True:
                    t0 = time.time()

                    combined_features.append(
                        np.histogram(lbp_img[y0_window:(y0_window + self.shape[0]),
                                             x0_window:(x0_window + self.shape[1])],
                                     bins=2**self._lbp_n_neighbors,
                                     range=(0, 2**self._lbp_n_neighbors))[0])

                    if DEBUG is True:
                        lbp_times.append(time.time() - t0)

                window_features.append(np.concatenate(combined_features))

                windows_.append(((x0_window_origin,
                                 y0_window_origin),
                                (x0_window_origin + window_shape[1],
                                 y0_window_origin + window_shape[0])))

                if DEBUG is True:
                    # Visualize the hog image and features
                    font_size = 16
                    if self.hog is True:
                        print("Time for HOG algorithm: {} s".
                              format(sum(hog_times)))

                        fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))

                        ax[0].imshow(img_gray[
                            y0_window_origin:y0_window_origin + window_shape[0],
                            x0_window_origin:x0_window_origin + window_shape[1]])
                        ax[0].set_title('Original image', fontsize=font_size)

                        ax[1].imshow(
                            hog_image[y0_window: y0_window + self.shape[0],
                                      x0_window: x0_window + self.shape[1]])
                        ax[1].set_title('HOG image', fontsize=font_size)

                        ax[2].plot(combined_features[0])
                        ax[2].set_title('hog features', fontsize=font_size)

                        plt.tight_layout()
                        plt.show()

                    # Visualize the lbp image and features
                    if self.lbp is True:
                        print("Time for LBP algorithm: {} s".
                              format(sum(lbp_times)))

                        fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))
                        ax[0].imshow(img_gray[
                            y0_window_origin:y0_window_origin + window_shape[0],
                            x0_window_origin:x0_window_origin + window_shape[1]])
                        ax[0].set_title('Original image', fontsize=font_size)

                        ax[1].imshow(lbp_img)
                        ax[1].set_title('LBP image', fontsize=font_size)
                        if self.hog is True:
                            ax[2].plot(combined_features[1])
                        else:
                            ax[2].plot(combined_features[0])
                        ax[2].set_title('LBP features', fontsize=font_size)

                        plt.tight_layout()
                        plt.show()

                x0_window += np.int(sliding_step[1]/scaling_factor[1])
                x0_window_origin += sliding_step[1]

            y0_window += np.int(sliding_step[0]/scaling_factor[0])
            y0_window_origin += sliding_step[0]

        return np.array(window_features, dtype=np.float32), windows_

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
        features_ = []
        for file in files:
            img = cv2.imread(file)
            assert img.dtype == np.uint8

            if img.shape != self.shape:
                img = cv2.resize(img, self.shape)

            window_features, _ = self.extract(img)

            if self.feature_shape is None:
                self.feature_shape = window_features[0].shape
            else:
                assert self.feature_shape == window_features[0].shape

            features_.append(window_features[0])
            if isinstance(max_files, int) and len(features_) > max_files:
                break

        return np.array(features_, dtype=np.float32)

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
    car_cls = CarClassifier(classifier=cls)

    car_cls.train(car_files, noncar_files, max_images=None)

    output = 'car_classifier.pkl'
    with open(output, "wb") as fp:
        pickle.dump(car_cls, fp)
        print("Car classifier was saved in {}".format(output))

    # Apply the classifier on a test image
    test_image = 'test_images/test_image_white_car.png'
    test_img = cv2.imread(test_image)

    features, windows = car_cls.extract(
        test_img, window_shape=(64, 64), sliding_step=(8, 8))

    predictions = car_cls.predict(features)

    car_cls.extract(test_img)
