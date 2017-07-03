#!/usr/bin/python

"""
CarClassifier class
"""
import numpy as np
import cv2

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from feature_extraction import HogExtractor, LbpExtractor


class CarClassifier(object):
    """A car classifier object"""
    def __init__(self, shape=(64, 64), classifier=None, extractor=None):
        """Initialization

        :param shape: tuple, 2x1
            Image shape in the training/testing data set.
        :param classifier: object
            Classifier object, e.g. LinearSVC(), DecisionTreeClassifier()
        :param extractor: object
            Feature extractor object, e.g. HogExtractor() or LbpExtractor()
        """
        if classifier is None:
            self.classifier = LinearSVC()  # default classifier
        else:
            self.classifier = classifier

        if extractor is None:
            self.extractor = HogExtractor()
        else:
            self.extractor = extractor

        # scaler is an attribute since it will be used by both
        # the train() and predict() methods.
        self.scaler = StandardScaler()

        self.shape = shape  # image shape

    def _load(self, files, max_files):
        """Read features from image files.

        :param: files: a list of string
            File names.
        :param max_files: int
            Maximum number of files to read.

        :returns : numpy.ndarray
            Features of images.
        """
        features = []
        for file in files:
            img = cv2.imread(file)
            assert img.dtype == np.uint8

            # Check the shape of image
            if img.shape[0:2] != self.shape:
                print("Warning: image shape is not {}".format(self.shape))
                img = cv2.resize(img, self.shape)

            # extract features
            img_features = self.extractor.extract(img)

            features.append(img_features)

            if len(features) > max_files:
                break

        return np.array(features, dtype=np.float32)

    def train(self, cars, non_cars, max_images=100000, test_size=0.3,
              random_state=None):
        """Train a car classifier.

        :param cars: list
            List of car filenames.
        :param non_cars: list
            List of non-car filenames.
        :param max_images: int
            Maximum number of file to read in each data set.
        :param test_size: float, in (0, 1)
            Percent of test data in self.X.
        :param random_state: int or None.
            Pseudo-random number generator state used for random sampling.
        """
        car_features = self._load(cars, max_images)
        noncar_features = self._load(non_cars, max_images)

        X = np.vstack((car_features, noncar_features))
        y = np.hstack((np.ones(len(car_features)),
                       np.zeros(len(noncar_features)))).astype(np.int8)

        X_shuffle, y_shuffle = shuffle(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_shuffle, y_shuffle, test_size=test_size, random_state=random_state)

        print("Number of training data: {}".format(len(y_train)))
        print("Number of testing data: {}".format(len(y_test)))
        print("Number of features: {}".format(len(X_train[0])))

        normalized_X = self.scaler.fit_transform(X_train)
        self.classifier.fit(normalized_X, y_train)

        y_pred = self._predict(X_test)
        print("Prediction accuracy on test set: {}".
              format(float(sum(y_pred == y_test) / len(y_pred))))

    def predict(self, img, binary=True):
        """Predict on a single image

        :param img: numpy.ndarray
            Image array.
        :param binary: Bool
            True for returning the binary result;
            False for returning the confidence score of the prediction.

        :return : numpy.array
            Class label or confidence score.
        """
        features = self.extractor.extract(img)

        if binary is False:
            return self._decision_function(np.array([features]))
        else:
            return self._predict(np.array([features]))

    def sliding_window_predict(self, img, step_size=(1.0, 1.0),
                               scale=1.0, binary=True):
        """Apply sliding window to an image and predict each window

        :param img: numpy.ndarray
            Image array.
        :param step_size: 1x2 tuple, float
            Size of the sliding step in the unit of the image shape.
        :param binary: Bool
            True for returning the binary result;
            False for returning the confidence score of the prediction.
        :param scale: float
            Scale of the original image.

        :return : numpy.array
            Class labels or confidence scores.
        """
        features, windows = self.extractor.sliding_window_extract(
            img, window_size=self.shape, step_size=step_size, scale=scale)

        if binary is False:
            return self._decision_function(features), windows
        else:
            return self._predict(features), windows

    def _decision_function(self, X):
        """Predict confidence scores for features (set)

        :param X: numpy.ndarray
            Features.

        :return: numpy.array
            Confidence scores.
        """
        return self.classifier.decision_function(self.scaler.transform(X))

    def _predict(self, X):
        """Predict class labels for features (set)

        :param X: numpy.ndarray
            Features.

        :return y_pred: numpy.array
            Class labels.
        """
        return self.classifier.predict(self.scaler.transform(X)).astype(np.int8)