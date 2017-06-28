"""
test class CarClassifier()
"""
import os
import glob
import pickle
import random
import cv2

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from feature_extraction import HogExtractor, LbpExtractor

from car_classifier import CarClassifier
from utilities import search_cars, draw_boxes


car_files = []
car_files.extend(glob.glob("data/vehicles/KITTI_extracted/*.png"))
car_files.extend(glob.glob("data/vehicles/GTI_Far/*.png"))
car_files.extend(glob.glob("data/vehicles/GTI_Left/*.png"))
car_files.extend(glob.glob("data/vehicles/GTI_Right/*.png"))
car_files.extend(glob.glob("data/vehicles/GTI_MiddleClose/*.png"))

noncar_files = []
noncar_files.extend(glob.glob("data/non-vehicles/Extras/*.png"))
noncar_files.extend(glob.glob("data/non-vehicles/GTI/*.png"))

output = 'car_classifier.pkl'
if not os.path.isfile(output):
    # Train a classifier
    cls = LinearSVC(C=0.0001)
    # cls = DecisionTreeClassifier(max_depth=10)
    # cls = RandomForestClassifier(n_estimators=20, max_depth=6)

    ext = HogExtractor(colorspace='YCrCb', cell_per_block=(2, 2))
    # ext = LbpExtractor(colorspace='YCrCb')

    # The critical hyper-parameter here is color_space='YCrCb'
    # A high accuracy (> 99%) is important here to reduce the
    # false-positive
    car_classifier = CarClassifier(classifier=cls, extractor=ext)
    car_classifier.train(
        car_files, noncar_files, test_size=0.2, max_images=10000)

    with open(output, "wb") as fp:
        pickle.dump(car_classifier, fp)
    print("Car classifier was saved in {}".format(output))
else:
    with open('car_classifier.pkl', "rb") as fp:
        car_classifier = pickle.load(fp)
    print("Load car classifier from {}".format(output))

# Test the classifier on a single image

fig, ax = plt.subplots(2, 4, figsize=(8, 4.5))
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

# Test sliding window classifier
with open('car_classifier.pkl', "rb") as fp:
    car_classifier = pickle.load(fp)

for i in range(10):
    test_image = 'test_images/test_image0{}.png'.format(i+1)
    test_img = cv2.imread(test_image)

    boxes = search_cars(test_img, car_classifier, scale_ratios=(0.5, 0.7),
                        confidence_thresh=0.2, overlap_thresh=0.2,
                        step_size=(0.125, 0.125), region=((0.0, 0.5), (1.0, 0.9)))

    test_img = draw_boxes(test_img, boxes)

    cv2.imshow('img', test_img)
    cv2.waitKey(0)