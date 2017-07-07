"""
test the lane line search pipe line
"""
import os
import pickle

import numpy as np
import cv2
import matplotlib.pyplot as plt

from camera_calibration import undistort_image
from threshold import thresh_image
from utilities import get_perspective_trans_matrix, two_plots
from parameters import project_video, thresh_params
from lane_line import LaneLine, sample_lines


image = "test_images/test_image04.png"

if os.path.isfile(image):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
else:
    raise OSError("{} does not exist!".format(image))

# -----------------------------------------------------------------------------
# undistort image
# -----------------------------------------------------------------------------

with open('camera_cali.pkl', "rb") as fp:
    camera_cali_ = pickle.load(fp)
test_img_undistorted = undistort_image(
    img, camera_cali_["obj_points"], camera_cali_["img_points"])

# -----------------------------------------------------------------------------
# perspective transform
# -----------------------------------------------------------------------------

src = project_video['src']
dst = project_video['dst']
ppt_trans_matrix, inv_ppt_trans_matrix = get_perspective_trans_matrix(src, dst)

warped = cv2.warpPerspective(
    img, ppt_trans_matrix, img.shape[:2][::-1])

warped = cv2.blur(warped, (15, 5), 0)

# Visualize the transform
img_ = np.copy(img)
cv2.polylines(img_, np.int32([src]), 1, (255, 255, 0), thickness=4)
two_plots(img_, warped, ('original', 'warped', 'check perspective transformation'))

# -----------------------------------------------------------------------------
# threshold
# -----------------------------------------------------------------------------

threshed = thresh_image(warped, thresh_params)

two_plots(warped, threshed, ('original', 'threshed', 'check threshing'))

# -----------------------------------------------------------------------------
# sample points belong to lines
# -----------------------------------------------------------------------------

points = sample_lines(threshed)

left = []
right = []
for i in range(len(points)):
    if points[i][0][0] < warped.shape[1]/2:
        left.append(i)
    elif points[i][0][0] > warped.shape[1]/2:
        right.append(i)

y_fit = np.arange(warped.shape[0], 0, -10)
left_line = LaneLine(y_fit)
right_line = LaneLine(y_fit)

if len(left) > 0:
    left_line.points = points[left[0]]
else:
    left_line.points = None

if len(right) > 0:
    right_line.points = points[right[-1]]
else:
    right_line.points = None

# Draw yellow lane lines in a black ground
warp_zero = np.zeros_like(warped).astype(np.uint8)
for line in (left_line, right_line):
    if line.x_fit is not None:
        plt.plot(line.points[0], line.points[1], 'r.')

        cv2.polylines(warped, np.int32([[np.vstack((line.x_fit, line.y_fit)).T]]),
                      0, (0, 0, 255), thickness=10)
        cv2.polylines(warp_zero, np.int32([[np.vstack((line.x_fit, line.y_fit)).T]]),
                      0, (0, 0, 255), thickness=30)

# Draw a green polygon in a black background
if left_line.p_fit is not None and right_line.p_fit is not None:
    pts = np.hstack(((left_line.x_fit, left_line.y_fit),
                     (right_line.x_fit[::-1], right_line.y_fit[::-1]))).T
    cv2.fillPoly(warp_zero, np.int_([pts]), (0, 255, 0))

plt.imshow(warped)
plt.show()

plt.imshow(warp_zero)
plt.show()

# Inverse perspective transformation
inv_warp = cv2.warpPerspective(warp_zero, inv_ppt_trans_matrix, warp_zero.shape[:2][::-1])

img_with_lanelines = cv2.addWeighted(img, 1.0, inv_warp, 0.5, 0.0)
plt.imshow(img_with_lanelines)
plt.show()