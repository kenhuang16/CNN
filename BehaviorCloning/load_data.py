import numpy as np
import random
from scipy import misc, ndimage
from sklearn.utils import shuffle
from matplotlib.colors import rgb_to_hsv


jitter_params = {'angle_jitter': 10,
          'angle_recover': 0.02,
          'shift_jitter': 60,
          'shift_recover': 0.008
          }


def batch_generator(X, y, batch_size, augment=False,
                    jitter_params=jitter_params):
    """Generate batched data.

    :param X: array-like
        Features.
    :param y: float
        Label.
    :param batch_size: int
        Size of batch.
    :param augment: Bool
        True for apply jitter to the features. Used for real time data
        augmentation (very time consuming).
    :params jitter_params: dict
        :key 'angle_jitter': float
            angle jitter in degs,
        :key 'angle_recover': float
            steering angle recovery,
        :key 'shift_jitter': float
            shift jitter in pixels,
        :key 'shift_recover': float
            steering angle recovery.

    :return: array like
        (Features, lables)
    """
    n = len(y)
    while 1:
        X_shuffled, y_shuffled = shuffle(X, y)
        for i in range(0, n, batch_size):
            X_batch = []
            y_batch = []
            for j in range(i, i + batch_size):
                if j >= n:
                    break

                if augment is False:
                    X_batch.append(preprocess(misc.imread(X_shuffled[j].strip())))
                    y_batch.append(y_shuffled[j])
                else:
                    # Generate random angle and shift jitters
                    shift_ = (2*random.random() - 1.0)*jitter_params['shift_jitter']
                    angle_ = (2*random.random() - 1.0)*jitter_params['angle_jitter']
                    X_jittered, y_jittered = \
                        jitter(misc.imread(X_shuffled[j].strip()), y_shuffled[j],
                               angle=angle_,
                               angle_recover=jitter_params['angle_recover'],
                               shift=shift_,
                               shift_recover=jitter_params['shift_recover'])

                    X_batch.append(preprocess(X_jittered))
                    y_batch.append(y_jittered)

            yield (np.array(X_batch), np.array(y_batch))


def preprocess(img):
    """Preprocessing an image

    Cropping, color space transformation and normalization.
    """
    img_ = img[60:126, 60:260]

    # img_ = misc.imresize(img_, 50)

    img_ = normalize_hsv(rgb_to_hsv(img_))
    assert abs(img_.max()) <= 1.0

    return img_


def jitter(X, y, angle=0.0, angle_recover=0, shift=0, shift_recover=0):
    """Apply rotation and/or shift jitter to an image

    The steering angle will be jittered accordingly.

    :param X: array-like
        Features (a single image).
    :param y: float
        Label.
    :param angle: float
        Rotation angle in degrees.
    :param shift: float
        Shift distance in pixels.

    :return:
        (features, label)
    """
    X_jittered = X[:, :, :]
    y_jittered = y

    if angle is not None:
        X_jittered = ndimage.interpolation.rotate(
            X_jittered, angle, reshape=False, order=2)
        y_jittered += angle*angle_recover

    if shift is not None:
        X_jittered = ndimage.interpolation.shift(X, [0, shift, 0], order=1)
        y_jittered += shift*shift_recover

    # Brightness and contrast jitter
    gain = 0.2  # Contrast is jittered between 1 +/- gain.
    bias = 30  # Bias is jittered between +/- bias.

    alpha = 1 + gain * (2 * random.random() - 1.0)
    beta = bias * (2 * random.random() - 1.0)
    X_jittered = alpha * X_jittered + beta
    X_jittered[X_jittered > 255] = 255
    X_jittered[X_jittered < 0] = 0

    # Important to change the dtype to np.uint8!!!
    return X_jittered.astype(np.uint8), y_jittered


def normalize_hsv(img):
    """Normalize an image with hsv channels"""
    img_ = np.copy(img)
    img_[:, :, 0] -= 0.5
    img_[:, :, 0] /= 0.5
    img_[:, :, 1] -= 0.5
    img_[:, :, 1] /= 0.5
    img_[:, :, 2] -= 127.5
    img_[:, :, 2] /= 127.5
    assert abs(img_.max()) <= 1.0

    return img_


# def rgb2gray(img):
#     """Convert RGB image to Gray scale"""
#     assert img.shape[2] == 3
#     img_gray = np.zeros([img.shape[0], img.shape[1], 1])
#     img_gray[:, :, 0] = \
#         0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
#
#     return img_gray