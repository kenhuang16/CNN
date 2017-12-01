"""
Train on Caltech256 dataset
"""
import os

from helper import maybe_download_caltech256
from data_processing import Caltech256
from model import train_generator, show_model, get_file_names
import keras_resnet


DEBUG = False


if __name__ == "__main__":
    input_shape = (224, 224, 3)
    batch_size = 8
    epochs = 120
    learning_rate = 1e-3

    root_path = os.path.expanduser('~/Projects/datasets')
    try:
        os.makedirs(root_path)
    except OSError:
        pass
    data_path = os.path.join(root_path, 'caltech256')
    maybe_download_caltech256(data_path)
    data = Caltech256(os.path.join(data_path, '256_ObjectCategories'),
                      n_trains=60)

    class_names = data.get_class_names()
    data.summary()

    model_file, weights_file, loss_history_file, model_txt_file = \
        get_file_names((50,), 'resnet')

    try:
        model = keras_resnet.build_resnet50_org(len(data.class_names))
        show_model(model, model_txt_file)
        gen_train = data.train_data_generator(input_shape, batch_size)
        steps_train = int(len(data.files_train)/batch_size)

        if DEBUG is True:
            epochs = 2
            steps_train = 2
            steps_vali = 2

        train_generator(model, gen_train, epochs, steps_train, learning_rate,
                        loss_history_file=loss_history_file,
                        weights_file=weights_file,
                        model_file=model_file)
    except PermissionError:
        pass