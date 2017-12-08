"""
Train on Caltech256 dataset
"""
import os

from data_processing import Caltech256
from model import train_generator, show_model, get_file_names
from networks import keras_resnet

DEBUG = False


if __name__ == "__main__":
    input_shape = (224, 224, 3)
    batch_size = 8
    epochs = 120
    learning_rate = 1e-3

    data_path = os.path.expanduser('~/Projects/datasets/caltech256/256_ObjectCategories')
    data = Caltech256(data_path, n_trains=60, seed=1)

    class_names = data.get_class_names()
    data.summary()

    model_file, weights_file, loss_history_file, model_txt_file = \
        get_file_names((50,), 'resnet', 'caltech256')

    try:
        model = keras_resnet.build_resnet50_org(len(data.class_names))
        show_model(model, model_txt_file)
        gen_train = data.train_data_generator(input_shape, batch_size)
        steps_train = int(len(data.image_files_train) / batch_size)
        gen_vali = data.train_data_generator(input_shape, batch_size, is_training=False)
        steps_vali = int(len(data.image_files_vali) / batch_size)

        if DEBUG is True:
            epochs = 2
            steps_train = 2
            steps_vali = 2

        train_generator(model, gen_train, epochs, steps_train, learning_rate,
                        gen_vali=gen_vali,
                        steps_vali=steps_vali,
                        loss_history_file=loss_history_file,
                        weights_file=weights_file,
                        model_file=model_file)
    except PermissionError:
        pass
