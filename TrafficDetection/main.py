"""
This is a combination of the advanced lane line finding and
the vehicle detection projects.

Run main.py to process the input video
"""
import os
import pickle

import matplotlib.pyplot as plt

from traffic import TrafficVideo
from parameters import project_video, test_video, thresh_params, \
                       car_search_params
from car_classifier import train_classifier


# Load the camera calibration parameters
camera_calibration_file='camera_cali.pkl'
if not os.path.isfile(camera_calibration_file):
    raise OSError(
        "Run unittest_camera_calibration.py to calibration the camera first!")

# Load the car classifier.
car_classifier_file = 'car_classifier.pkl'
if not os.path.isfile(car_classifier_file):
    raise OSError("Run unittest_car_classifier.py to train a classifier!")

video = project_video
# video = test_video

# Process the video
ppt_trans_params = (video['frame'], video['src'], video['dst'])

# Process the video
f1 = TrafficVideo(video['input'], camera_cali_file='camera_cali.pkl',
                  perspective_trans_params=ppt_trans_params,
                  thresh_params=thresh_params,
                  car_classifier_file=car_classifier_file,
                  car_search_params=car_search_params,
                  is_search_laneline=False,
                  is_search_car=True)

# f1.show_perspective_transform(100)

f1.process(video['output'])

# for i in range(100, 2500, 100):
#     print("Frame {}".format(i))
#     image = f1.get_video_image(i)
#     plt.imsave('test_images/test_image{:02d}.png'.format(int(i/100)), image)
#     fig, ax = plt.subplots()
#     ax.imshow(image)
#     plt.tight_layout()
#     plt.show()
