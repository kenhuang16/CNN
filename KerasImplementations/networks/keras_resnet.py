"""
The ResNet models in the paper

[Deep Residual Learning for Image Recognition]
(http://arxiv.org/abs/1512.03385)

implemented in Keras.
"""
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Add
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras import regularizers
from keras.initializers import he_normal


BLOCK_NAMES = [chr(x) for x in range(ord('a'), ord('z') + 1)]
BLOCK_NAMES.extend([chr(x) for x in range(ord('A'), ord('Z') + 1)])


def identity_block(X, n_filters, stage, block, weight_decay,
                   is_first_stage=False,
                   is_first_stage_layer=False):
    """Implementation of the identity block

    :param X: input tensor with shape (None, n_H_prev, n_W_prev, n_C_prev)
    :param n_filters: integer
        No. of filters in the Convolutional layers.
    :param stage: integer
        Used to name the layers.
    :param block: integer
        Used to name the layers.
    :param weight_decay: float
        Strength of L2 regularization.
    :param is_first_stage: bool
    :param is_first_stage_layer: bool
        The first layer of a stage (except for the first stage after
        the 3x3 max-pooling layer) has a stride of 2 which reduces the
        dimensions (w and h) both by a factor of two. In the meanwhile,
        the shortcut layer should also follow a 1x1 convolutional layer
        with a stride of two plus a batch normalization layer.

    :return X: output tensor with shape (None, n_H, n_W, n_C)
    """
    conv_name_base = 'res' + str(stage) + BLOCK_NAMES[block] + '_branch'
    bn_name_base = 'bn' + str(stage) + BLOCK_NAMES[block] + '_branch'

    # The shortcut path is branch 1 and the main path is branch 2

    X_shortcut = X

    # First component of main path
    strides = (1, 1)
    if is_first_stage is False and is_first_stage_layer is True:
        strides = (2, 2)
        X_shortcut = Conv2D(filters=n_filters, kernel_size=(1, 1), strides=(2, 2),
                            kernel_regularizer=regularizers.l2(weight_decay),
                            name=conv_name_base + '1',
                            kernel_initializer=he_normal())(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=strides, padding='same',
               name=conv_name_base + '2a',
               kernel_regularizer=regularizers.l2(weight_decay),
               kernel_initializer=he_normal())(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu', name=conv_name_base + '2a_relu')(X)

    # Second component of main path
    X = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name=conv_name_base + '2b',
               kernel_regularizer=regularizers.l2(weight_decay),
               kernel_initializer=he_normal())(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu', name=conv_name_base + '2b_relu')(X)

    # Add shortcut value to main path
    merge_name_base = 'res' + str(stage) + BLOCK_NAMES[block]
    X = Add(name=merge_name_base)([X, X_shortcut])
    X = Activation('relu', name=merge_name_base + '_relu')(X)

    return X


def bottleneck_block(X, n_filters, stage, block, weight_decay,
                     is_first_stage=False,
                     is_first_stage_layer=False):
    """Implementation of the bottleneck block

    :param X: input tensor with shape (None, n_H_prev, n_W_prev, n_C_prev)
    :param n_filters: integer
        No. of filters in the convolutional layers.
    :param stage: integer
        Used to name the layers.
    :param block: integer
        Used to name the layers.
    :param weight_decay: float
        Strength of L2 regularization.
    :param is_first_stage: bool
    :param is_first_stage_layer: bool
        The first layer of a stage (except for the first stage after
        the 3x3 max-pooling layer) has a stride of 2 which reduces the
        dimensions (w and h) both by a factor of two. In the meanwhile,
        the shortcut layer should also follow a 1x1 convolutional layer
        with a stride of two plus a batch normalization layer.

    :return X: output tensor with shape (None, n_H, n_W, n_C)
    """
    conv_name_base = 'res' + str(stage) + BLOCK_NAMES[block] + '_branch'
    bn_name_base = 'bn' + str(stage) + BLOCK_NAMES[block] + '_branch'

    # The shortcut path is branch 1 and the main path is branch 2
    X_shortcut = X

    # First component of main path
    strides = (1, 1)
    # Note that it is different from the identity block
    if is_first_stage_layer is True:
        if is_first_stage is True:
            strides = (2, 2)

        X_shortcut = Conv2D(filters=n_filters * 4, kernel_size=(1, 1),
                            strides=strides, name=conv_name_base + '1',
                            kernel_regularizer=regularizers.l2(weight_decay),
                            kernel_initializer=he_normal())(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Conv2D(filters=n_filters, kernel_size=(1, 1), strides=strides,
               name=conv_name_base + '2a',
               kernel_regularizer=regularizers.l2(weight_decay),
               kernel_initializer=he_normal())(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu', name=conv_name_base + '2a_relu')(X)

    # Second component of main path
    X = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name=conv_name_base + '2b',
               kernel_regularizer=regularizers.l2(weight_decay),
               kernel_initializer=he_normal())(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu', name=conv_name_base + '2b_relu')(X)

    # Third component of main path
    X = Conv2D(filters=4*n_filters, kernel_size=(1, 1), strides=(1, 1),
               name=conv_name_base + '2c',
               kernel_regularizer=regularizers.l2(weight_decay),
               kernel_initializer=he_normal())(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = Activation('relu', name=conv_name_base + '2c_relu')(X)

    # Add shortcut value to main path
    merge_name_base = 'res' + str(stage) + BLOCK_NAMES[block]
    X = Add(name=merge_name_base)([X, X_shortcut])
    X = Activation('relu', name=merge_name_base + '_relu')(X)

    return X


def build_resnet(input_shape, num_classes, stage_blocks, n_filters=64,
                 weight_decay=1e-4, bottleneck=True,
                 first_kernels=(3, 3), first_strides=(1, 1),
                 max_pool_sizes=None, max_pool_strides=None):
    """Build the ResNet

    :param input_shape: tuple
        Shape of the input tensor.
    :param num_classes: integer
        Number of classes.
    :param stage_blocks: array-like
        Number of repetitions of block in each stage.
    :param n_filters: integer
        Number of filters in the starting stage. The number of filters
        increase by a factor of two when advancing to the next stage.
    :param weight_decay: float
        Strength of L2 regularization.
    :param bottleneck: bool
        True for using the bottleneck block and False for using the
        identity block.
    :param first_kernels: int / tuple
        Kernel sizes of the first convolutional layer.
    :param first_strides: int / tuple
        Strides of the first convolutional layer.
    :param max_pool_sizes: None / int / tuple
        Pool size of the max-pooling layer.
    :param max_pool_strides: None / int / tuple
        Strides of the max-pooling layer.

    :return: keras model
    """
    inputs = Input(input_shape, name='input')

    # entrance block
    X = Conv2D(filters=n_filters, kernel_size=first_kernels,
               strides=first_strides, padding='same', name='conv1',
               kernel_regularizer=regularizers.l2(weight_decay),
               kernel_initializer=he_normal())(inputs)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu', name='conv1_relu')(X)
    if max_pool_sizes is not None:
        X = MaxPooling2D(pool_size=max_pool_sizes, strides=max_pool_strides,
                         padding='same', name='max_pool')(X)

    # repeated residual network blocks
    for i in range(len(stage_blocks)):
        for j in range(stage_blocks[i]):
            is_first_stage = False
            is_first_stage_layer = False
            if i == 0:
                is_first_stage = True
            if j == 0:
                is_first_stage_layer = True

            stage_idx = i + 2
            block_idx = j

            if bottleneck is True:
                X = bottleneck_block(X, n_filters, stage_idx, block_idx, weight_decay,
                                     is_first_stage, is_first_stage_layer)
            else:
                X = identity_block(X, n_filters, stage_idx, block_idx, weight_decay,
                                   is_first_stage, is_first_stage_layer)

        n_filters *= 2

    # classifier block
    X = GlobalAveragePooling2D(name='avg_pool')(X)
    outputs = Dense(num_classes, activation='softmax', name='fc',
                    kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_initializer=he_normal())(X)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def build_resnet18_org(num_classes, weight_decay=1e-4):
    """ResNet18

    :param num_classes: int
        Number of classes to classify.
    :param weight_decay: float
        Strength of L2 regularization.
    """
    return build_resnet((224, 224, 3), num_classes, (2, 2, 2, 2),
                        weight_decay=weight_decay, bottleneck=False,
                        first_kernels=(7, 7), first_strides=(2, 2),
                        max_pool_sizes=(3, 3), max_pool_strides=(2, 2))


def build_resnet34_org(num_classes, weight_decay=1e-4):
    """ResNet34

    :param num_classes: int
        Number of classes to classify.
    :param weight_decay: float
        Strength of L2 regularization.
    """
    return build_resnet((224, 224, 3), num_classes, (3, 4, 6, 3),
                        weight_decay=weight_decay, bottleneck=False,
                        first_kernels=(7, 7), first_strides=(2, 2),
                        max_pool_sizes=(3, 3), max_pool_strides=(2, 2))


def build_resnet50_org(num_classes, weight_decay=1e-4):
    """ResNet50

    :param num_classes: int
        Number of classes to classify.
    :param weight_decay: float
        Strength of L2 regularization.
    """
    return build_resnet((224, 224, 3), num_classes, (3, 4, 6, 3),
                        weight_decay=weight_decay, bottleneck=True,
                        first_kernels=(7, 7), first_strides=(2, 2),
                        max_pool_sizes=(3, 3), max_pool_strides=(2, 2))


def build_resnet101_org(num_classes, weight_decay=1e-4):
    """ResNet101

    :param num_classes: int
        Number of classes to classify.
    :param weight_decay: float
        Strength of L2 regularization.
    """
    return build_resnet((224, 224, 3), num_classes, (3, 4, 23, 3),
                        weight_decay=weight_decay, bottleneck=True,
                        first_kernels=(7, 7), first_strides=(2, 2),
                        max_pool_sizes=(3, 3), max_pool_strides=(2, 2))


def build_resnet152_org(num_classes, weight_decay=1e-4):
    """ResNet152

    :param num_classes: int
        Number of classes to classify.
    :param weight_decay: float
        Strength of L2 regularization.
    """
    return build_resnet((224, 224, 3), num_classes, (3, 8, 36, 3),
                        weight_decay=weight_decay, bottleneck=True,
                        first_kernels=(7, 7), first_strides=(2, 2),
                        max_pool_sizes=(3, 3), max_pool_strides=(2, 2))


if __name__ == "__main__":
    # ResNet50 (1000 classes) should have 25.6 M parameters
    model = build_resnet50_org(1000)
    model.summary()
