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

        self.feature_shape = None  # feature shape

    def _load(self, files, max_num_files):
        """Read features from image files.

        :param: files: a list of string
            File names.
        :param max_num_files: int
            Maximum number of files to read.

        :returns : numpy.ndarray
            Features of images.
        """
        features = []
        for file in files:
            img = cv2.imread(file)
            assert img.dtype == np.uint8

            if img.shape[0:2] != self.shape:
                print("Warning: image shape is not {}".format(self.shape))
                img = cv2.resize(img, self.shape)

            img_features, _ = self.extractor.extract(img)

            # Get the shape of the feature
            if self.feature_shape is None:
                self.feature_shape = img_features.shape
            else:
                assert self.feature_shape == img_features.shape

            features.append(img_features)
            if len(features) > max_num_files:
                break

        return np.array(features, dtype=np.float32)

    def train(self, cars, non_cars, max_images=100000, test_size=0.3,
              random_state=None):
        """Train a car classifier.

        :param cars: list
            List of car file names.
        :param non_cars: list
            List of non-car file names.
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

        t0 = time.time()

        normalized_X = self.scaler.fit_transform(X_train)
        self.classifier.fit(normalized_X, y_train)
        print("Training finished in {:.1f} s".format(time.time() - t0))

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
        features, _ = self.extractor.extract(img)

        if binary is False:
            return self._decision_function(np.array([features]))
        else:
            return self._predict(np.array([features]))

    def sliding_window_predict(self, img, step_size=None, scale=(1.0, 1.0),
                               binary=True):
        """Apply sliding window to an image and predict each window

        :param img: numpy.ndarray
            Image array.
        :param step_size: 1x2 tuple, int
            Size of the sliding step.
        :param binary: Bool
            True for returning the binary result;
            False for returning the confidence score of the prediction.
        :param scale: 1x2 tuple, float
            Scale of the original image.

        :return : numpy.array
            Class labels or confidence scores.
        """
        if step_size is None:
            step_size = self.shape

        features, windows = self.extractor.sliding_window_extract(
            img, window_size=self.shape, step_size=step_size, scale=scale)

        if binary is False:
            return self._decision_function(features), windows
        else:
            return self._prediction(features), windows

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


if __name__ == "__main__":
    case = 3

    # Train a classifier
    if case == 1:
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

        ext = HogExtractor(colorspace='YCrCb')
        # The critical hyper-parameter here is color_space='YCrCb'
        # A high accuracy (> 99%) is important here to reduce the
        # false-positive
        car_cls = CarClassifier(classifier=cls, extractor=ext)

        car_cls.train(car_files, noncar_files, test_size=0.2, max_images=5000)

        output = 'car_classifier.pkl'
        with open(output, "wb") as fp:
            pickle.dump(car_cls, fp)
            print("Car classifier was saved in {}".format(output))

    # Test the classifier on a single image
    elif case == 2:
        with open('car_classifier.pkl', "rb") as fp:
            car_classifier = pickle.load(fp)

        car_image = "data/vehicles/KITTI_extracted/1.png"
        car_img = cv2.imread(car_image)

        plt.imshow(car_img)
        plt.show()

        predictions = car_classifier.predict(car_img)
        print(predictions)

        noncar_image = "data/non-vehicles/GTI/image1.png"
        noncar_img = cv2.imread(noncar_image)

        plt.imshow(noncar_img)
        plt.show()

        predictions = car_classifier.predict(noncar_img)
        print(predictions)

    # Test sliding window classifier
    elif case == 3:
        with open('car_classifier.pkl', "rb") as fp:
            car_classifier = pickle.load(fp)

        test_image = 'test_images/test_image_white_car.png'
        test_img = cv2.imread(test_image)

        predictions, windows = car_classifier.sliding_window_predict(
            test_img, step_size=(16, 16), binary=False, scale=(0.25, 0.25))

        for window, prediction in zip(windows, predictions):
            if prediction > 0.0:
                cv2.rectangle(test_img, window[0], window[1], (0, 0, 255), 6)

        plt.imshow(test_img)
        plt.show()
