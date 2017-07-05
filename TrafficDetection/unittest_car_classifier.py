"""
test class CarClassifier()
"""
import os
import pickle
import random
import cv2

import matplotlib.pyplot as plt

from car_classifier import train_classifier
from utilities import sw_search_car, draw_box, read_image_data, augment_image_data
from parameters import car_files, noncar_files, car_search_params


# -----------------------------------------------------------------------------
# Visualize the augmented images
# -----------------------------------------------------------------------------

imgs, labels = read_image_data()
imgs_augmented, labels_augmented = augment_image_data(imgs, labels, 16)
fig, ax = plt.subplots(4, 4, figsize=(8, 9))
ax = ax.flatten()
for i in range(len(ax)):
    ax[i].imshow(imgs_augmented[i])
    ax[i].set_title("Label: {}".format(labels_augmented[i]))
    ax[i].set_axis_off()

plt.suptitle("Augmented images")
plt.show()

# -----------------------------------------------------------------------------
# Load or train a classifier
# -----------------------------------------------------------------------------

output = 'car_classifier.pkl'
if not os.path.isfile(output):
    train_classifier(output, 5000)

with open('car_classifier.pkl', "rb") as fp:
    car_classifier = pickle.load(fp)
print("Load car classifier from {}".format(output))

# -----------------------------------------------------------------------------
# Test the classifier on a single image
# -----------------------------------------------------------------------------

_, ax = plt.subplots(2, 4, figsize=(8, 4.5))
ax = ax.flatten()
for i in range(len(ax)):
    if i < 4:
        image = random.choice(car_files)
    else:
        image = random.choice(noncar_files)
    img = cv2.imread(image)
    prediction = car_classifier.predict(img)

    ax[i].imshow(img)
    ax[i].set_title("Prediction: {}".format(prediction), fontsize=10)
plt.show()

# -----------------------------------------------------------------------------
# Test sliding window classifier
# -----------------------------------------------------------------------------

with open('car_classifier.pkl', "rb") as fp:
    car_classifier = pickle.load(fp)

for i in range(10):
    test_image = 'test_images/test_image{:02d}.png'.format(i+1)
    test_img = cv2.imread(test_image)

    boxes = sw_search_car(
        test_img, car_classifier,
        step_size=car_search_params['step_size'],
        scales=car_search_params['scales'],
        regions=car_search_params['regions'],
        confidence_thresh=car_search_params['confidence_thresh'],
        overlap_thresh=car_search_params['overlap_thresh'],
        heat_thresh=car_search_params['heat_thresh'])

    test_img = draw_box(test_img, boxes)

    cv2.imshow('img', test_img)
    cv2.waitKey(0)
