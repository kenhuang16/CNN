"""

"""
import numpy as np
from skimage.transform import resize

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape
from keras.optimizers import Adam
from keras import backend as K


K.set_image_data_format('channels_last')


def dice_coef(y_true, y_pred):
    """"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    """"""
    return -dice_coef(y_true, y_pred)


class Preprocessor(object):
    """Data preprocessor"""
    def __init__(self, shape):
        """Initialization

        :param shape: tuple
            Data shape after preprocessing.
        """
        self._mean = None
        self._shape = shape

    def resize(self, X):
        """Resize the data"""
        X_resized = []
        for i in range(X.shape[0]):
            X_resized.append(resize(X[i, :, :], self._shape, mode='reflect'))

        return np.asarray(X_resized)

    def transform(self, X, subtract_mean=True, normalize=True):
        """Transform the data"""
        X_new = np.copy(X)
        if subtract_mean is True:
            X_new = X - self._mean
        if normalize is True:
            X_new /= 255

        X_resized = self.resize(X_new)

        if len(X_resized.shape) == 4:
            return X_resized
        if len(X_resized.shape) == 3:
            # Add an axis to the gray-scale image input
            return np.expand_dims(X_resized, 3)
        else:
            raise ValueError("Wrong data shape: {}!".format(X_new.shape))

    def fit(self, X):
        """Fit the data"""
        self._mean = np.mean(X)
        self._std = np.std(X)

    def fit_transform(self, X):
        """Apply transform after fit"""
        self.fit(X)

        return self.transform(X)

class UNet(object):
    """U-net class"""
    def __init__(self, input_shape):
        """Initialization

        :param input_shape: tuple, (width, height)
            Shape of the input layer.
        """
        self.input_shape = input_shape
        self.model = None
        self._construct()

        self.preprocessor = Preprocessor(input_shape)

    def _construct(self):
        """Construct the network"""
        input_layer = Input((self.input_shape[0], self.input_shape[1], 1))
        conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
        conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

        conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2_1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

        conv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3_1)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_2)

        conv4_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4_1)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_2)

        conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5_1)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2),
                                           padding='same')(conv5_2), conv4_2], axis=3)
        conv6_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6_1)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2),
                                           padding='same')(conv6_2), conv3_2], axis=3)
        conv7_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7_1)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2),
                                           padding='same')(conv7_2), conv2_2], axis=3)
        conv8_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8_1)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2),
                                           padding='same')(conv8_2), conv1_2], axis=3)
        conv9_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9_1)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9_2)

        self.model = Model(inputs=[input_layer], outputs=[conv10])

        self.model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=[dice_coef])

        self.model.summary()
