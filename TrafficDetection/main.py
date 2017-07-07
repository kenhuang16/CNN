"""
This is a combination of the advanced lane line finding and
the vehicle detection projects.

Run main.py to process the input video
"""
import os

from traffic import TrafficVideo
from parameters import project_video, test_video, thresh_params, \
                       car_search_params


# Load the camera calibration parameters
camera_calibration_file = 'camera_cali.pkl'
if not os.path.isfile(camera_calibration_file):
    raise OSError(
        "Run unittest_camera_calibration.py to calibration the camera first!")

# Load the car classifier.
car_classifier_file = 'car_classifier.pkl'
if not os.path.isfile(car_classifier_file):
    raise OSError("Run unittest_car_classifier.py to train a classifier!")

video = project_video
# video = test_video
#
# Process the video
ppt_trans_params = (video['src'], video['dst'])

# Process the video
f1 = TrafficVideo(video['input'], camera_cali_file='camera_cali.pkl',
                  perspective_trans_params=ppt_trans_params,
                  thresh_params=thresh_params,
                  car_classifier_file=car_classifier_file,
                  car_search_params=car_search_params,
                  is_search_laneline=True,
                  is_search_car=False)

f1.process(video['output'])
