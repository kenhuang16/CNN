"""
Functions:
    - read_image_data()
    - augment_image_data()
"""
import random

import numpy as np
import cv2


def read_image_data(car_files, noncar_files, max_count=100000):
    """Read image data from files.

    :param max_count: int
        Maximum number of data to read.

    :returns: numpy.ndarray
        Image data.
    :return labels: 1D numpy.ndarray
        Image labels.
    """
    imgs = []
    labels = np.hstack((np.ones(len(car_files)),
                        np.zeros(len(noncar_files)))).astype(np.int8)

    for file in car_files + noncar_files:
        img = cv2.imread(file)
        imgs.append(img)

        if len(imgs) > max_count:
            break

    return np.array(imgs, dtype=np.uint8), labels


def augment_image_data(imgs, labels, number=5000, gain=0.3, bias=30):
    """Augment image data

    @param imgs: numpy.ndarray
        Features.
    @param labels: 1D numpy.ndarray
        Labels.
    @param number: int
        Number of augmented data.
    @param gain: float
        Maximum gain jitter.
    @param bias: int
        Maximum absolute bias jitter.

    @return new_features: numpy.ndarray
        Augmented features.
    @return new_labels: 1D numpy.ndarray
        Augmented labels.
    """
    new_imgs = []
    new_labels = []

    size = len(imgs)
    for i in range(number):
        choice = random.randint(0, size - 1)
        img = imgs[choice]

        # Randomly flip the image horizontally
        if random.random() > 0.5:
            img = cv2.flip(img, 1)

        # Randomly remove the color information
        if random.random() > 0.5:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.stack([img_gray, img_gray, img_gray], axis=-1)

        # Random brightness and contrast jitter
        if img.mean() > 155:
            alpha = 1 + gain*(random.random() - 1.0)
            beta = bias*(random.random() - 1.0)
        elif img.mean() < 100:
            alpha = 1 + gain*(1.0 - random.random())
            beta = bias*(1.0 - random.random())
        else:
            alpha = 1 + gain * (2 * random.random() - 1.0)
            beta = bias * (2 * random.random() - 1.0)
        img = img*alpha + beta
        img[img > 255] = 255
        img[img < 0] = 0

        new_imgs.append(img)
        new_labels.append(labels[choice])

    return np.asarray(new_imgs, dtype=np.uint8), np.asarray(new_labels)


