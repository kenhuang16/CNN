"""
Apply ResNet on the CIFAR-10 data set
"""
import numpy as np
from keras.datasets import cifar10

from model import train, evaluate, show_model, get_file_names
from networks.keras_resnet import build_resnet

if __name__ == "__main__":
    learning_rate = 1e-3
    epochs = 100
    batch_size = 128

    # Load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    num_classes = len(np.unique(y_train))
    assert(num_classes == len(np.unique(y_test)))
    input_shape = X_train.shape[1:]  # (32, 32, 3)

    print("Number of classes: {}".format(num_classes))
    print("Input data shape: {}".format(input_shape))
    print("Number of train data: {}".format(len(X_train)))
    print("Number of test data: {}".format(len(y_test)))

    # Build the model
    model_blocks = (3, 3, 3)
    model_file, weights_file, loss_history_file, model_txt_file = get_file_names(
        model_blocks, 'resnet_small')

    try:
        model = build_resnet(input_shape, num_classes, model_blocks,
                             n_filters=16, bottleneck=False)
        show_model(model, model_txt_file)
        train(model, X_train, y_train, epochs, batch_size, learning_rate,
              loss_history_file=loss_history_file,
              weights_file=weights_file,
              model_file=model_file)
    except PermissionError:
        pass

    metrics = evaluate(X_test, y_test, model_file, weights_file, batch_size=128)
    print("Loss on the test set: {}".format(metrics[0]))
    print("Accuracy on the test set: {}".format(metrics[1]))
