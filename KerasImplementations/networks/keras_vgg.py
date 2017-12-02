"""
The VGG models in the paper

[Very Deep Convolutional Networks for Large-scale Image Recognition]
(https://arxiv.org/pdf/1409.1556.pdf)

implemented in Keras.
"""
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras import regularizers
from keras.initializers import he_normal
from keras.models import Model


LAYER_NAMES = [chr(x) for x in range(ord('a'), ord('z') + 1)]


def build_vgg(input_shape, num_classes, block_layers, fc_layers, n_filters=64,
              weight_decay=5e-4, dropout_probas=None, batch_normalization=True):
    """Build VGG style network

    :param input_shape: tuple
        Shape of the input tensor.
    :param num_classes: integer
        Number of classes.
    :param block_layers: array-like
        Number of repetitions of block in each stage.
    :param fc_layers: array-like
        Size of each fully-connected layers
    :param n_filters: integer
        Number of filters in the starting block. The number of filters
        increase by a factor of two when advancing to the next block,
        but is upper-bounded by 512.
    :param weight_decay: float
        Strength of L2 regularization.
    :param dropout_probas: array-like
        Dropout probabilities in each dropout layers. None for without
        dropout layers. Note, dropout layers will be placed after
        the first two fully-connected layers each if dropout_probas
        is not None.
    :param batch_normalization: bool
        True for include a batch normalization layers after each
        convolutional layer.

    :return: model in Keras
    """
    inputs = Input(input_shape)
    X = inputs

    for i in range(len(block_layers)):
        for j in range(block_layers[i]):
            block_idx = str(i + 1)
            X = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same',
                       name='conv_' + block_idx + LAYER_NAMES[j])(X)
            if batch_normalization is True:
                X = BatchNormalization(
                    axis=3, name='bn_' + block_idx + LAYER_NAMES[j])(X)
            X = Activation(
                'relu', name='conv_' + block_idx + LAYER_NAMES[j] + '_relu')(X)

        X = MaxPooling2D(pool_size=(2, 2), padding='valid',
                         name='max_pool_' + str(i))(X)
        if n_filters <= 256:
            n_filters *= 2

    X = Flatten(name='flatten')(X)

    for i, n_layers in enumerate(fc_layers):
        fc_idx = str(i+1)
        X = Dense(n_layers, activation='relu',
                  kernel_regularizer=regularizers.l2(weight_decay),
                  kernel_initializer=he_normal(),
                  name='fc_' + fc_idx)(X)
        if dropout_probas is not None:
            X = Dropout(dropout_probas[i], name='fc_' + fc_idx + '_dropout')(X)

    outputs = Dense(num_classes, activation='softmax',
                    kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_initializer=he_normal(),
                    name='fc_' + str(len(fc_layers)+1))(X)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def build_vgg16_org(num_classes):
    """VGG16"""
    return build_vgg((224, 224, 3), num_classes, (2, 2, 3, 3, 3), (4096, 4096),
                     n_filters=64, weight_decay=5e-4, dropout_probas=(0.5, 0.5))


def build_vgg19_org(num_classes):
    """VGG19"""
    return build_vgg((224, 224, 3), num_classes, (2, 2, 4, 4, 4), (4096, 4096),
                     n_filters=64, weight_decay=5e-4, dropout_probas=(0.5, 0.5))


if __name__ == "__main__":
    # VGG16 (1000 classes) should have 138.4 M parameters
    model = build_vgg16_org(1000)
    model.summary()

    model = build_vgg((80, 80, 3), 100, (2, 3, 4), (512, 512),
                        n_filters=64, weight_decay=5e-4,
                        dropout_probas=None,
                        batch_normalization=False)
    model.summary()
