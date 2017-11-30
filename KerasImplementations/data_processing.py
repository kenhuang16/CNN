"""
TODO: data augmentation for training

Now I simply resize the image on the shorter side and randomly crops
a 224 x 224 part.
"""
import random
import numpy as np
import os
import abc

import glob
import cv2


def shuffle_data(X, y, seed=None):
    """Randomly shuffle the data

    :param X: numpy.ndarray
        Features.
    :param y: numpy.ndarray
        Labels.
    :param seed: None / int
        For random number generation.

    :return: shuffled features and labels
    """
    s = list(range(X.shape[0]))
    random.seed(seed)
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


def crop_image(img, size):
    """"""
    ratio = img.shape[0] / img.shape[1]
    if ratio > 1:
        img = cv2.resize(img, (size, int(size*ratio)))
        i_start = random.randint(0, img.shape[0] - size)
        return img[i_start:(i_start + size), :]
    else:
        img = cv2.resize(img, (int(size/ratio), size))
        i_start = random.randint(0, img.shape[1] - size)
        return img[:, i_start:(i_start + size)]


def preprocess_data(X_files, labels, input_shape, num_classes, indices=None,
                    is_training=True):
    """Data cropping, augmentation and normalization

    :param X_files: array of strings
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
        True for data augmentation.

    :return X: numpy.ndarray, (None, w, h, c)
        Preprocessed features.
    :return y: numpy.ndarray, (None, num_classes)
        One-hot encoded labels.
    """
    if indices is None:
        indices = list(range(len(labels)))

    X = np.empty((len(indices), *input_shape), dtype=float)
    for i, idx in enumerate(indices):
        X[i, :, :, :] = crop_image(cv2.imread(X_files[idx]), input_shape[0])
        normalize_rgb_images(X)
    y = convert_to_one_hot(labels[indices], num_classes)
    return X, y


class Caltech(abc.ABC):
    """Caltech dataset abstract class"""
    def __init__(self, data_path, n_trains=30):
        """Initialization

        :param data_path: string
            Path of the data folder.
        :param n_trains: int
            Number of training data per class.
        """
        self.data_path = data_path
        self.class_names = self.get_class_names()

        self.files_train = None  # image files' full paths
        self.labels_train = None  # y label (indices of the classes)
        self.files_test = None
        self.labels_test = None
        self.images_per_class = None
        self.split_data(n_trains)

    @abc.abstractmethod
    def get_class_names(self):
        """Get class names in a list"""
        pass

    def split_data(self, n_trains):
        """Split data into train, validation and test set

        :param n_trains: int
            Number of training data per class.
        """
        files_train = []
        labels_train = []
        files_test = []
        labels_test = []
        images_per_class = []
        for idx, dir_name in enumerate(self.class_names):
            imgs = glob.glob(os.path.join(self.data_path, dir_name, '*.jpg'))
            images_per_class.append(len(imgs))
            random.seed(idx)  # use fixed seed here
            random.shuffle(imgs)
            labels = [idx]*len(imgs)

            files_train.extend(imgs[:n_trains])
            labels_train.extend(labels[:n_trains])

            files_test.extend(imgs[n_trains:])
            labels_test.extend(labels[n_trains:])

        self.files_train = np.array(files_train)
        self.labels_train = np.array(labels_train)
        self.files_test = np.array(files_test)
        self.labels_test = np.array(labels_test)
        self.images_per_class = np.array(images_per_class)

    def summary(self):
        """Print the summary of the dataset"""
        print("Number of classes: {}".format(len(self.class_names)))
        print("Minimum, maximum and median No. images per class: {}, {}, {}".
              format(self.images_per_class.min(),
                     self.images_per_class.max(),
                     int(np.median(self.images_per_class))))
        print("Number of training data: {}".format(len(self.files_train)))
        print("Number of test data: {}".format(len(self.files_test)))

    def data_generator(self, input_shape, batch_size, category):
        """Batch data generator

        :param input_shape: tuple, (w, h, c)
            Input shape for the neural network.
        :param batch_size: int
            Batch size.
        :param category: string
            'train' or 'test'.

        :return: batches of (images, labels)
        """
        is_training = False
        if category == 'train':
            X_files = self.files_train
            labels = self.labels_train
            is_training = True
        elif category == 'test':
            X_files = self.files_test
            labels = self.labels_test
        else:
            raise ValueError("Unknown category! Must be 'train' or 'test'")

        n = int(len(X_files) / batch_size)
        while 1:
            X_files, labels = shuffle_data(X_files, labels)
            for i in range(n):
                indices = [i*batch_size + j for j in range(batch_size)]
                X, y = preprocess_data(
                    X_files, labels, input_shape, len(self.class_names),
                    indices, is_training)

                yield X, y


class Caltech101(Caltech):
    """Caltech101 dataset class"""
    def get_class_names(self):
        """Get class names in a list"""
        class_names = sorted([x for x in os.listdir(self.data_path)
                              if x != 'BACKGROUND_Google'])
        assert(len(class_names) == 101)
        return class_names


class Caltech256(Caltech):
    """Caltech256 dataset class"""
    def get_class_names(self):
        """Get class names in a list"""
        class_names = sorted([x for x in os.listdir(self.data_path)
                              if x != '257.clutter'])
        assert (len(class_names) == 256)
        return class_names
