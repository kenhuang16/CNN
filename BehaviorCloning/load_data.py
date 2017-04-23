import numpy as np
import random
from scipy import misc, ndimage
from sklearn.utils import shuffle
from matplotlib.colors import rgb_to_hsv


class BatchGenerator(object):
    def __init__(self, X, y, batch_size, preprocessor=None):
        """Initialize object.

        :param X: Nx2 numpy array.
            The first column is the path of the image file and the
            second column is the 'ref' indicator.
        :param y: float
            Label.
        :param batch_size: int
            Size of batch.
        :param preprocessor: None/object
            BatchPreprocessor instance.
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.preprocessor = preprocessor

    def fit(self):
        """Generate batched data

        :return: features and labels.
        """
        n = len(self.y)
        while 1:
            X_shuffled, y_shuffled = shuffle(self.X, self.y)
            for i in range(0, n, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size].tolist()
                y_batch = y_shuffled[i:i+self.batch_size].tolist()

                for j in range(len(y_batch)):
                    X_batch[j], y_batch[j] = \
                        self.preprocessor.fit(X_batch[j], y_batch[j])

                yield (np.array(X_batch), np.array(y_batch))


class Preprocessor(object):
    def __init__(self, preprocess_params, jitter_params=None, drive=False):
        """
        :param preprocess_params: dict
            Parameters used in preprocessing.
            :key channels: string
                Channels..
            :key size: int, float or tuple
                int - Percentage of current size.
                float - Fraction of current size.
                tuple - Size of the output image.

        :params jitter_params: None/dict
            If not None, apply jitter to the features. Used for real time data
            augmentation (very time consuming).

            :key 'angle_jitter': float
                Maximum angle jitter in degs.
            :key 'angle_recover': float
                steering angle recovery. When the car is heading towards
                the edge of the road, apply a correction angle to make it
                go straight.
            :key 'shift_jitter': float
                Maximum shift jitter in pixels,
            :key 'shift_recover': float
                steering angle recovery. When the car is near the edge of
                the road, apply a small correction angle to make it return
                to the center of the road. Note that if a large recover
                angle is applied here, the car will wiggle violently.
            :key gain: float
                Contrast is jittered between 1 +/- gain.
            :key bias: int/float
                Bias is jittered between +/- bias.

        :param drive: Bool
            True for used in drive.py. Default is False.
        """
        self._size = preprocess_params['size']
        self._channels = preprocess_params['channels']

        if jitter_params is not None:
            self._angle_jitter = jitter_params['angle_jitter']
            self._angle_recover = jitter_params['angle_recover']
            self._shift_jitter = jitter_params['shift_jitter']
            self._shift_recover = jitter_params['shift_recover']
            self._gain = jitter_params['gain']
            self._bias = jitter_params['bias']
        else:
            self._angle_jitter = None
            self._angle_recover = None
            self._shift_jitter = None
            self._shift_recover = None
            self._gain = None
            self._bias = None

        if self._gain is None:
            self._gain = 0.0
        if self._bias is None:
            self._bias = 0.0

        self.drive = drive

    def fit(self, X, y=None):
        """Preprocessing an image

        :param X: numpy array.
            If self.drive is True
                Image array.
            else
                The first element is the path of the image file and the
                second element is the 'ref' indicator.
        :param y: None/float
            Steering angle.

        :return X_new: numpy array
            Image array.
        :return y_new: float
            New steering angle.

        """
        if self.drive is True:
            X_new = np.copy(X)
            y_new = True
        else:
            file_path = X[0].strip()
            ref = X[1]

            X = misc.imread(file_path)

            # Apply jitter
            if y is None or ref == 1:
                X_new = np.copy(X)
                y_new = None
                if ref == 1:
                    y_new = y
            else:
                X_new, y_new = self._jitter(X, y)

        # crop
        X_new = X_new[60:126, 60:260]

        # scale
        X_new = misc.imresize(X_new, self._size)

        # normalize
        if self._channels == 'rgb':
            X_new = self.normalize_rgb(X_new)
        elif self._channels == 'hsv':
            X_new = self.normalize_hsv(X_new)
        else:
            raise ValueError("Unknown channels!")

        return X_new, y_new

    def _jitter(self, X, y):
        """Apply rotation and/or shift jitter to an image

        :param X: numpy array
            Image array.
        :param y: float
            Steering angle.

        :return X_jittered: numpy array
            Jittered image array.
        :return y_jittered: float
            Jittered steering angle.
        """
        X_jittered = np.copy(X)
        y_jittered = np.copy(y)

        # 50% probability to flip the image
        if random.random() > 0.5:
            X_jittered = np.fliplr(X_jittered)
            y_jittered *= -1.0

        if self._angle_jitter is not None:
            angle_random = (2 * random.random() - 1.0)
            X_jittered = ndimage.interpolation.rotate(
                X_jittered, angle_random*self._angle_jitter,
                reshape=False, order=2)
            y_jittered += angle_random * self._angle_recover

        if self._shift_jitter is not None:
            shift_random = (2 * random.random() - 1.0)
            X_jittered = ndimage.interpolation.shift(
                X_jittered, [0, shift_random*self._shift_jitter, 0], order=1)
            y_jittered += shift_random * self._shift_recover

        # Brightness and contrast jitter
        alpha = 1 + self._gain * (2 * random.random() - 1.0)
        beta = self._bias * (2 * random.random() - 1.0)
        X_jittered = alpha * X_jittered + beta
        X_jittered[X_jittered > 255] = 255
        X_jittered[X_jittered < 0] = 0

        # Important to change the dtype to np.uint8!!!
        return X_jittered.astype(np.uint8), y_jittered

    @staticmethod
    def normalize_rgb(img):
        """Normalize an RGB image

        :param img: np.ndarray
            Image array

        :return Normalized image array.
        """
        img_ = np.copy(img)
        img_ = img_.astype(np.float)/255.0
        img_ -= 0.5

        assert abs(img_.max()) <= 1.0

        return img_

    @staticmethod
    def normalize_hsv(img):
        """Normalize an image with hsv channels.

        :param img: np.ndarray
            Image array.

        :return Normalized image array.
        """
        img_ = rgb_to_hsv(img)
        img_[:, :, 0] -= 0.5
        img_[:, :, 0] /= 0.5
        img_[:, :, 1] -= 0.5
        img_[:, :, 1] /= 0.5
        img_[:, :, 2] -= 127.5
        img_[:, :, 2] /= 127.5
        assert abs(img_.max()) <= 1.0

        return img_
