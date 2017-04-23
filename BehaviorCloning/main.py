import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
from keras.models import load_model

from set_model import set_model
from load_data import BatchGenerator, Preprocessor


saved_model = 'cnn_network.h5'

batch_size = 128

# -------------

angle_correction = 0.13  # correction angle for left (+) and right (-) images

# shift_jitter does not help here
jitter_params = {
    'angle_jitter': 15,
    'angle_recover': 0.15,
    'shift_jitter': None,
    'shift_recover': 0.10,
    'gain': None,
    'bias': None
}
# One must modify preprocess parameters in drive.py
# -------------

model_params = {
    'name': 'NVIDIA',
    'input_shape': (66, 200, 3)
}
preprocess_params = {
    'channels': 'hsv',
    'size': 1.0
}

# model_params = {
#     'name': None,
#     'input_shape': (33, 100, 3)
# }
# preprocess_params = {
#     'channels': 'rgb',
#     'size': 0.5
# }

# Read the data log file
data_files = ["data/driving_log_3.csv", "data/driving_log_4.csv"]

drive_data = []
for data_file in data_files:
    new_data = pd.read_csv(
        data_file, header=0,
        names=['center_image', 'left_image', 'right_image',
               'steering_angle', 'throttle', 'break', 'speed'])
    drive_data.append(new_data)

drive_data = pd.concat(drive_data)

# Only the path of the image file and steering angle are required.
#
# A new column 'ref' is added. It can be used to control the center,
# left and right images separately in real time augmentation.
drive_data_center = drive_data[['center_image', 'steering_angle']]
drive_data_center['ref'] = 1
drive_data_center = drive_data_center.as_matrix()

drive_data_left = drive_data[['left_image', 'steering_angle']]
drive_data_left['ref'] = 0
drive_data_left = drive_data_left.as_matrix()
drive_data_left[:, 1] += angle_correction

drive_data_right = drive_data[['right_image', 'steering_angle']]
drive_data_right['ref'] = 0
drive_data_right = drive_data_right.as_matrix()
drive_data_right[:, 1] -= angle_correction

drive_data = np.concatenate([drive_data_center, drive_data_left, drive_data_right])
# X data contains both the path name of the image file and
# the 'ref' indicator.
X_drive_data = drive_data[:, [0, 2]]
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
preprocessor_train = Preprocessor(
    jitter_params=jitter_params, preprocess_params=preprocess_params)
generator_train = BatchGenerator(
    X_train, y_train, batch_size, preprocessor=preprocessor_train).fit()

preprocessor_vali = Preprocessor(preprocess_params=preprocess_params)
generator_vali = BatchGenerator(
    X_train, y_train, batch_size, preprocessor=preprocessor_vali).fit()

try:
    cnn_network = load_model(saved_model)
    cnn_network.load_weights(saved_model)
    print("\nSuccessfully loaded existing model!")
except IOError:
    print("\nCannot find existing model!")
    print("\nStart training new model!")

history = cnn_network.fit_generator(
    generator=generator_train,
    steps_per_epoch=int(len(y_train)/batch_size),
    epochs=1,
    validation_data=generator_vali,
    validation_steps=int(len(y_vali)/batch_size))

cnn_network.save(saved_model)
print("Model saved!")
