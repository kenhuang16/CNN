"""
test camera calibration and image undistortion
"""
import glob
import cv2
import pickle

from camera_calibration import calibrate_camera, undistort_image
from utilities import two_plots


chess_board_images_ = glob.glob('camera_calibration_images/calibration*.jpg')
pattern_size_ = (9, 6)
camera_cali_file = "camera_cali.pkl"

calibrate_camera(
    chess_board_images_, pattern_size=pattern_size_, output=camera_cali_file)

test_img = cv2.imread("camera_calibration_images/calibration1.jpg")
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

with open(camera_cali_file, "rb") as fp:
    camera_cali_ = pickle.load(fp)
test_img_undistorted = undistort_image(
    test_img, camera_cali_["obj_points"], camera_cali_["img_points"])

two_plots(test_img, test_img_undistorted,
          ('original', 'undistorted', 'camera calibration'), output='')
