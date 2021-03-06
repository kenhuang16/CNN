"""
TODO: add validation data set (20% of the training data set)
"""
import random
import numpy as np
import os
import abc

import glob
import cv2


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def shuffle_data(X, y):
    """Randomly shuffle the data

    :param X: numpy.ndarray
        Features.
    :param y: numpy.ndarray
        Labels.

    :return: shuffled features and labels
    """
    s = list(range(X.shape[0]))
    random.shuffle(s)
    return X[s], y[s]


def normalize_rgb_images(imgs):
    """Normalize an RGB image data in-place

    :param imgs: numpy.ndarray
        Image data. It can have the shape (None, w, h, ch) or (w, h, ch)

    :return: normalized image data.
    """
    imgs /= 127.5
    imgs -= 1.0


def convert_to_one_hot(y, num_classes):
    """Apply the one-hot encoding

    :param y: array like with the shape (None, class index)
        Original y label.
    :param num_classes: int
        Number of classes

    :return: numpy.array
        One-hot encoded labels.
    """
    return np.eye(num_classes)[y.reshape(-1)]


def crop_image(img, crop_size, is_training=True):
    """Crop the image to a square image with given size.

    :param img: numpy.ndarray
        Input image
    :param crop_size: int
        Size (w and h) of the cropped image.
    :param is_training: bool
        True for training data and False for validation data.

    :return: numpy.ndarray
        Cropped image.
    """
    # First scale the original image to three different sizes
    if is_training is True:
        scale_size = random.choice([int(crop_size*1.1),
                                    int(crop_size*1.3),
                                    int(crop_size*1.5)])
    else:
        scale_size = crop_size

    # resize to make the shorter size equal to 'scale_size'
    ratio = img.shape[0] / img.shape[1]
    if ratio > 1:
        img = cv2.resize(img, (scale_size, int(scale_size*ratio)))
    else:
        img = cv2.resize(img, (int(scale_size/ratio), scale_size))

    if is_training is True:
        # random crop a part with a size (crop_size, crop_size)
        w_start = random.randint(0, img.shape[1] - crop_size)
        h_start = random.randint(0, img.shape[0] - crop_size)
    else:
        # crop the central part of the image
        w_start = int((img.shape[1] - crop_size) / 2)
        h_start = int((img.shape[0] - crop_size) / 2)

    return img[h_start:(h_start + crop_size),
               w_start:(w_start + crop_size)]


def flip_horizontally(img):
    """Flip the image horizontally

    :param img: numpy.ndarray
        Input image.

    :return: numpy.ndarray
        Flipped image.
    """
    return np.flip(img, axis=0)


def preprocess_training_data(image_files, labels, input_shape, num_classes,
                             indices=None, is_training=True):
    """Data cropping, augmentation and normalization

    :param image_files: array of strings
        File paths.
    :param labels: array of integers
        Class indices.
    :param input_shape: tuple, (w, h, c)
        Input shape for the neural network.
    :param num_classes: int
        Number of classes.
    :param indices: array-like / None
        Indices of the batch data (None for the whole input data).
    :param is_training: bool
        True for training data and False for validation data.
    :return X: numpy.ndarray, (None, w, h, c)
        Preprocessed features.
    :return y: numpy.ndarray, (None, num_classes)
        One-hot encoded labels.
    """
    if indices is None:
        indices = list(range(len(labels)))

    X = np.empty((len(indices), *input_shape), dtype=float)
    for i, idx in enumerate(indices):
        X[i, :, :, :] = crop_image(cv2.imread(image_files[idx]), input_shape[0],
                                   is_training=is_training)
        if random.random() > 0.5:
            flip_horizontally(X)

    normalize_rgb_images(X)
    y = convert_to_one_hot(labels[indices], num_classes)
    return X, y


class Caltech(abc.ABC):
    """Caltech dataset abstract class"""
    def __init__(self, data_path, n_trains=30, n_tests=None,
                 validation_train_split=0.2, seed=None):
        """Initialization

        :param data_path: string
            Path of the data folder.
        :param n_trains: int
            Number of training data per class.
        :param n_tests: int / None
            Number of test data per class. If None, then the rest data
            are used for test.
        :param validation_train_split: float
            Percentage of training images used for validation.
        :param seed: int
            Seed used for data split.
        """
        self.data_path = data_path
        self.class_names = self.get_class_names()

        self.image_files_train = None  # image files' full paths
        self.labels_train = None  # y label (indices of the classes)
        self.image_files_vali = None
        self.labels_vali = None
        self.image_files_test = None
        self.labels_test = None
        self.images_per_class = None
        self.split_data(n_trains, n_tests, validation_train_split, seed=seed)

    @abc.abstractmethod
    def get_class_names(self):
        """Get class names in a list"""
        pass

    def split_data(self, n_trains, n_tests, validation_train_split, seed=None):
        """Split data into train, validation and test set

        :param n_trains: int
            Number of training data per class.
        :param n_tests: int / None
            Number of test data per class. If None, then the rest data
            are used for test.
        :param validation_train_split: int
            Percentage of training images used for validation.
        :param seed: int
            Seed used for data split.
        """
        image_files_train = []
        labels_train = []
        image_files_vali = []
        labels_vali = []
        image_files_test = []
        labels_test = []
        images_per_class = []

        random.seed(seed)  # fix data splitting

        n_valis = int(validation_train_split*n_trains)
        for idx, dir_name in enumerate(self.class_names):
            image_files = glob.glob(os.path.join(self.data_path, dir_name, '*.jpg'))
            images_per_class.append(len(image_files))
            random.shuffle(image_files)
            labels = [idx]*len(image_files)

            image_files_vali.extend(image_files[:n_valis])
            labels_vali.extend(labels[:n_valis])
            image_files_train.extend(image_files[n_valis:n_trains])
            labels_train.extend(labels[n_valis:n_trains])

            if n_tests is None:
                image_files_test.extend(image_files[n_trains:])
                labels_test.extend(labels[n_trains:])
            else:
                n_total = n_trains + n_tests
                image_files_test.extend(image_files[n_trains:n_total])
                labels_test.extend(labels[n_trains:n_total])

        self.image_files_train = np.array(image_files_train)
        self.labels_train = np.array(labels_train)
        self.image_files_vali = np.array(image_files_vali)
        self.labels_vali = np.array(labels_vali)
        self.image_files_test = np.array(image_files_test)
        self.labels_test = np.array(labels_test)
        self.images_per_class = np.array(images_per_class)

        random.seed(None)  # reset seed

    def summary(self):
        """Print the summary of the dataset"""
        print("Number of classes: {}".format(len(self.class_names)))
        print("Minimum, maximum and median No. images per class: {}, {}, {}".
              format(self.images_per_class.min(),
                     self.images_per_class.max(),
                     int(np.median(self.images_per_class))))
        print("Number of training data: {}".format(len(self.image_files_train)))
        print("Number of validation data: {}".format(len(self.image_files_vali)))
        print("Number of test data: {}".format(len(self.image_files_test)))

    def train_data_generator(self, input_shape, batch_size, is_training=True):
        """Batch training data generator

        The data will be randomly resized, cropped and then flipped
        horizontally.

        :param input_shape: tuple, (w, h, c)
            Input shape for the neural network.
        :param batch_size: int
            Batch size.
        :param is_training: bool
            True for training data and False for validation data.

        :return: batches of (images, labels)
        """
        if is_training is True:
            files = self.image_files_train
            labels = self.labels_train
        else:
            files = self.image_files_vali
            labels = self.labels_vali

        n = int(len(files) / batch_size)
        while 1:
            files_shuffled, labels_shuffled = shuffle_data(files, labels)
            for i in range(n):
                indices = [i*batch_size + j for j in range(batch_size)]
                X, y = preprocess_training_data(
                    files_shuffled, labels_shuffled, input_shape, len(self.class_names),
                    indices, is_training=is_training)

                yield X, y


class Caltech101(Caltech):
    """Caltech101 dataset class"""
    def get_class_names(self):
        """Get class names in a list"""
        class_names = sorted([x for x in os.listdir(self.data_path)
                              if x != 'BACKGROUND_Google'])
        return class_names


class Caltech256(Caltech):
    """Caltech256 dataset class"""
    def get_class_names(self):
        """Get class names in a list"""
        class_names = sorted([x for x in os.listdir(self.data_path)
                              if x != '257.clutter'])
        return class_names
