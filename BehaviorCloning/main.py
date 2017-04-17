import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
from keras.models import load_model

from set_model import set_model
from load_data import batch_generator


saved_model = 'cnn_network.h5'

c_run = True  # Flag for continuous run
batch_size = 128

angle_correction = 0.1  # correction angle for left (+) and right (-) images

# -------------
# One must modify preprocess function for different input shape
# -------------

model_params = {
    'name': 'NVIDIA',
    'input_shape': (66, 200, 3)
}

# model_params = {
#     'name': None,
#     'input_shape': (33, 100, 3)
# }

# Read the data log file
data_files = ["driving_log_1.csv", "driving_log_2.csv",
              "driving_log_3.csv", "driving_log_4.csv"]

drive_data = []
for data_file in data_files:
    new_data = pd.read_csv(
        "data/" + data_file, header=0,
        names=['center_image', 'left_image', 'right_image',
               'steering_angle', 'throttle', 'break', 'speed'])
    drive_data.append(new_data)

drive_data = pd.concat(drive_data)

# Only the file name and steering angle are required
drive_data_center = drive_data[['center_image', 'steering_angle']].as_matrix()

drive_data_left = drive_data[['left_image', 'steering_angle']].as_matrix()
drive_data_left[:, 1] += angle_correction

drive_data_right = drive_data[['right_image', 'steering_angle']].as_matrix()
drive_data_right[:, 1] -= angle_correction

drive_data = np.concatenate([drive_data_center, drive_data_left, drive_data_right])
X_drive_data = drive_data[:, 0]
y_drive_data = drive_data[:, 1]

X_train, X_vali, y_train, y_vali = \
    train_test_split(X_drive_data, y_drive_data, test_size=0.2)

print("Number of train data: {}, number of validation data: {}.".
      format(len(X_train), len(y_vali)))

# ------------------
# Set up the network

cnn_network = set_model(model_params)

cnn_network.summary()

adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)
cnn_network.compile(optimizer=adam, loss='mean_squared_error')

# -----------------
# Train the network

if os.path.isfile(saved_model) and c_run is False:
    print("{} already exist!".format(saved_model))
else:
    if c_run is True:
        # Fine tuning the existing model
        try:
            cnn_network = load_model(saved_model)
            cnn_network.load_weights(saved_model)
            print("Successfully loaded existing model!")
        except IOError:
            print("Cannot find existing model!")

    history = cnn_network.fit_generator(
        generator=batch_generator(X_train, y_train, batch_size, augment=True),
        steps_per_epoch=int(len(y_train)/batch_size),
        epochs=2,
        validation_data=batch_generator(X_vali, y_vali, batch_size),
        validation_steps=int(len(y_vali)/batch_size))

    cnn_network.save(saved_model)
    print("Model saved!")
