from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras import initializers


def set_model(params):
    """Set a Keras Sequential model.

    :param params: dictionary
        :key 'name': string
            Name of the network,
            - "None" for a simple testing network,
            - "NVIDIA" for the network published at https://arxiv.org/abs/1604.07316
            "End to end learning for self-driving cars".
        :key 'shape': tuple
            Shape of the input layer

    :return:
        Keras Sequential model.
    """
    init = initializers.random_normal(stddev=0.1)

    cnn_network = Sequential()

    if params['name'] is None:
        cnn_network.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 2),
                               kernel_initializer=init, padding='same',
                               input_shape=params['input_shape']))
        cnn_network.add(Activation('relu'))
        cnn_network.add(MaxPooling2D(pool_size=(2, 2)))

        cnn_network.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2),
                               kernel_initializer=init, padding='same'))
        cnn_network.add(Activation('relu'))
        cnn_network.add(MaxPooling2D(pool_size=(2, 2)))

        cnn_network.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                               kernel_initializer=init, padding='same'))
        cnn_network.add(Activation('relu'))
        cnn_network.add(MaxPooling2D(pool_size=(2, 2)))

        cnn_network.add(Flatten())

        cnn_network.add(Dropout(rate=0.5))

        cnn_network.add(Dense(128, kernel_initializer=init))
        cnn_network.add(Activation('relu'))

        cnn_network.add(Dropout(rate=0.5))

        cnn_network.add(Dense(64, kernel_initializer=init))
        cnn_network.add(Activation('relu'))

        # For regression problem
        cnn_network.add(Dense(1))
        cnn_network.summary()
    elif params['name'] == 'NVIDIA':
        cnn_network.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2),
                               kernel_initializer=init, padding='valid',
                               input_shape=params['input_shape']))
        cnn_network.add(Activation('relu'))

        cnn_network.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2),
                               kernel_initializer=init, padding='valid'))
        cnn_network.add(Activation('relu'))

        cnn_network.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2),
                               kernel_initializer=init, padding='valid'))
        cnn_network.add(Activation('relu'))

        cnn_network.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                               kernel_initializer=init, padding='valid'))
        cnn_network.add(Activation('relu'))

        cnn_network.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                               kernel_initializer=init, padding='valid'))
        cnn_network.add(Activation('relu'))

        cnn_network.add(Flatten())

        cnn_network.add(Dropout(rate=0.5))

        cnn_network.add(Dense(100, kernel_initializer=init))
        cnn_network.add(Activation('relu'))

        cnn_network.add(Dropout(rate=0.5))

        cnn_network.add(Dense(50, kernel_initializer=init))
        cnn_network.add(Activation('relu'))

        # For regression problem
        cnn_network.add(Dense(1))
    else:
        raise ValueError("Unknown model: {}".format(params['name']))

    return cnn_network