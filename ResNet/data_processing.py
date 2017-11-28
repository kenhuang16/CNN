import random
import numpy as np


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
    """Normalize an RGB image data

    :param imgs: numpy.ndarray
        Image data. It can have the shape (None, w, h, ch) or (w, h, ch)

    :return: normalized image data.
    """
    imgs = imgs.astype(np.float32) / 127.5
    imgs -= 1.0
    return imgs


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
