#!/usr/bin/python
"""
"""
import os
import pickle
import glob
import numpy as np
import cv2
from scipy import misc

from moviepy.editor import VideoFileClip

from traffic import TrafficVideo
from calibration import calibrate_camera, undistort_image
from image_transform import calibrate_perspective_trans, perspective_trans
import matplotlib.pyplot as plt
from my_plot import double_plot

from traffic_classifiers import CarClassifier
from parameters import project_video, thresh_params


my_video = project_video

input = my_video['input']
output = my_video['output']
i_frame = my_video['i_frame']
source_points = my_video['src']
distortion_points = my_video['dst']

# -----------------------------------------------------------------------------
# Camera calibration
camera_cali_file = "camera_cali.pkl"
if not os.path.isfile(camera_cali_file):
    chess_board_images = glob.glob('camera_cal/calibration*.jpg')
    pattern_size = (9, 6)
    calibrate_camera(
        chess_board_images, pattern_size=pattern_size, output=camera_cali_file)

# Apply un-distortion on a test image
clip = VideoFileClip(input)
test_img = clip.get_frame(my_video['i_frame'] / 25.0)
# misc.imsave('test_images/test_img_perspective_trans.png', test_img)
with open(camera_cali_file, "rb") as fp:
    camera_cali = pickle.load(fp)
test_img_undistorted = undistort_image(
    test_img, camera_cali["obj_points"], camera_cali["img_points"])

# -----------------------------------------------------------------------------
# Perspective transformation
perspective_trans_file = "perspective_trans.pkl"
calibrate_perspective_trans(source_points, distortion_points,
                            output=perspective_trans_file)

# Apply perspective transformation on a test image
warped = perspective_trans(test_img_undistorted, perspective_trans_file)

# Visualize the transformation
cv2.polylines(test_img_undistorted, np.int32([source_points]),
              1, (255, 255, 0), thickness=4)
cv2.polylines(warped, np.int32([distortion_points]),
              1, (255, 255, 0), thickness=4)
double_plot(test_img_undistorted, warped,
            ('original', 'warped', 'perspective transformation'), output='')

# -----------------------------------------------------------------------------
# Process the video
f1 = TrafficVideo(input, camera_cali_file=camera_cali_file,
                  perspective_trans_file=perspective_trans_file,
                  thresh_params=thresh_params,
                  car_classifier="car_classifier.pkl"
)

for i in np.arange(1, 2500, 10):
    print("Frame {}".format(i))
    image = f1.process_video_image(frame=i)
    #misc.imsave('test_images/test_img.png', test_img)
    fig, ax = plt.subplots()
    ax.imshow(image)
    plt.tight_layout()
    plt.show()

f1.process(output)