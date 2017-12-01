import os
import pickle
import collections

from contextlib import redirect_stdout
from keras.optimizers import Adam
from keras.models import model_from_json

from data_processing import convert_to_one_hot, normalize_rgb_images


def get_file_names(model_blocks, root_name, dir_name='save'):
    """Get model, weights and loss history files

    :param model_blocks:
    :param root_name: string
        Root name of these files.
    :param dir_name: string
        Files' directory.

    :return: model_file, weights_file, loss_history_file
    """
    model_basename = root_name + '_' + '-'.join(map(str, model_blocks))
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    loss_history_file = os.path.join(dir_name, model_basename + '_loss.pkl')
    weights_file = os.path.join(dir_name, model_basename + '_weights.h5')
    model_file = os.path.join(dir_name, model_basename + '_model.json')
    model_txt_file = os.path.join(dir_name, model_basename + '_model.txt')

    return model_file, weights_file, loss_history_file, model_txt_file


def show_model(model, filename):
    """Printout and save the model to a txt file

    :param model: Keras model
    :param filename: String
        Name of the text file.
    """
    model.summary()
    with open(filename, 'w') as f:
        with redirect_stdout(f):
            model.summary()


def save_model(model, file_path):
    """Save Keras model to file

    :param model: Keras model
    :param file_path: string
        Path of the output file.
    """
    model_json = model.to_json()
    with open(file_path, "w") as f:
        f.write(model_json)
    print("Saved model to file!")


def load_model(file_path):
    """Load model from file

    :param file_path: string
        Path of the model file.
    :return: Keras model
    """
    with open(file_path, 'r') as f:
        loaded_model_json = f.read()
        model = model_from_json(loaded_model_json)
        print("Loaded model from file!")
        return model


def save_weights(model, file_path):
    """Save trained weights to file

    :param model: Keras model
    :param file_path: string
        Path of the output file.
    """
    if file_path is not None:
        model.save(file_path)
    print("Saved weights to file!")


def save_history(history, file_path):
    """Save loss (other metrics) history to file

    :param history: dictionary
        Output of Model.fit() in Keras.
    :param file_path: string
        Path of the output file.
    """
    if file_path is not None:
        loss_history = dict()
        for key in history.history.keys():
            loss_history[key] = list()

        if os.path.exists(file_path):
            with open(file_path, "rb") as fp:
                loss_history = pickle.load(fp)

        for key in history.history.keys():
            loss_history[key].extend(history.history[key])

        with open(file_path, "wb") as fp:
            pickle.dump(loss_history, fp)
    print("Saved training history to file!")


def train(model, X, y, epochs, batch_size, learning_rate,
          loss_history_file=None, weights_file=None, model_file=None):
    """Train the model

    :param model: keras.models.Model object
        Keras model.
    :param X: numpy.ndarray
        Features.
    :param y: numpy.ndarray
        Labels.
    :param epochs: int
        Number of epochs.
    :param batch_size: int
        Batch size.
    :param learning_rate: float
        Learning rate.
    :param loss_history_file: string
        File name for loss history.
    :param weights_file: string
        File name for storing the weights of the model.
    :param model_file: string
        File name for the storing the model.
    """
    try:
        model.load_weights(weights_file)
        print("\nLoaded weights from file!")
    except OSError:
        print("\nStart training new model!")

    model.compile(optimizer=Adam(learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    normalize_rgb_images(X.astype(float))
    num_classes = model.layers[-1].output_shape[-1]
    y = convert_to_one_hot(y, num_classes)

    history = model.fit(X, y, batch_size=batch_size, epochs=epochs,
                        shuffle=True, validation_split=0.1)

    save_history(history, loss_history_file)
    save_model(model, model_file)
    save_weights(model, weights_file)


def train_generator(model, gen, epochs, steps_train, learning_rate,
                    loss_history_file=None, weights_file=None, model_file=None):
    """Train the model by feeding data generators

    :param model: keras.models.Model object
        Keras model.
    :param gen: generator
        Train data generator.
    :param epochs: int
        Number of epochs.
    :param steps_train: int
        Number of batches per epoch for train data generator.
    :param learning_rate: float
        Learning rate.
    :param loss_history_file: string
        File name for loss history.
    :param weights_file: string
        File name for storing the weights of the model.
    :param model_file: string
        File name for the storing the model.
    """
    try:
        model.load_weights(weights_file)
        print("\nLoaded weights from file!")
    except:
        print("\nStart training new model!")

    model.compile(optimizer=Adam(learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit_generator(gen, epochs=epochs, steps_per_epoch=steps_train)

    save_history(history, loss_history_file)
    save_model(model, model_file)
    save_weights(model, weights_file)


def evaluate(X, y, model_file, weights_file, batch_size=32):
    """Evaluate a data set.

    :param X: numpy.ndarray
        Features.
    :param y: numpy.ndarray
        Labels.
    :param model_file: string
        Path of the saved JSON file.
    :param weights_file: string
        Path of the HDF5 file.
    :param batch_size: int
        Batch size for evaluation. Default is 32.

    :return: list of loss and other metrics.
    """
    model = load_model(model_file)
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(weights_file)

    normalize_rgb_images(X)
    num_classes = model.layers[-1].output_shape[-1]
    y = convert_to_one_hot(y, num_classes)

    metrics = model.evaluate(X, y, batch_size=batch_size)
    if isinstance(metrics, collections.Iterable):
        return metrics
    return [metrics]
