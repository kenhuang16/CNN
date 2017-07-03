#!/usr/bin/python
"""
"""
import pickle
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

from lane_line import TwinLine
from threshold import Threshold
from utilities import draw_boxes, search_cars


INF = 1.0e21
# The length (in pixel) of the front of the car in the original image.
CAR_FRONT_LENGTH = 40

DEBUG = False


class TrafficVideo(object):
    """TrafficVideo class"""
    def __init__(self, input, camera_cali_file=None, perspective_trans_file=None,
                 thresh_params=None, max_poor_fit_time=0.5, car_classifier=None,
                 search_car=True, search_laneline=True):
        """Initialization.

        :param input: string
            File name of the input video.
        :param camera_cali_file: string
            Name of the pickle file storing the camera calibration
        :param perspective_trans_file: string
            Name of the pickle file storing the perspective transform
            matrices.
        :param thresh_params: list of dictionary
            Parameters for the gradient and color threshhold.
        :param max_poor_fit_time: float
            Maximum allowed period (in second) of consecutive fail before
            a fresh line search.
        :param car_classifier:
            Car classifier pickle file.
        :param search_car: Boolean
            True for search cars in the video/image.
        :param search_laneline: Boolean
            True for search lanelines in the video/image.
        """
        self.input = input
        self.clip = VideoFileClip(input)
        self.iframe = 0
        self._shape = None  # image (in the video) shape

        self.lines = None

        with open(camera_cali_file, "rb") as fp:
            camera_cali = pickle.load(fp)
        self.obj_points = camera_cali["obj_points"]
        self.img_points = camera_cali["img_points"]
        self.camera_matrix = None
        self.dist_coeffs = None

        with open(perspective_trans_file, "rb") as fp:
            perspective = pickle.load(fp)
        self.perspective_matrix = perspective["matrix"]
        self.inv_perspective_matrix = perspective["inv_matrix"]

        self.thresh_params = thresh_params

        self.max_poor_fit_time = max_poor_fit_time

        if thresh_params is not None and search_laneline is True:
            self._is_search_laneline = True
        else:
            self._is_search_laneline = False

        if car_classifier is not None and search_car is True:
            try:
                with open(car_classifier, "rb") as fp:
                    self.car_classifier = pickle.load(fp)
            except IOError:
                raise IOError("Not found: car classifier!")

            self._is_search_car = True
        else:
            self._is_search_car = False

    @property
    def shape(self):
        """The shape property"""
        return self._shape

    @shape.setter
    def shape(self, value):
        """"""
        if self._shape != value[0:2]:
            self._shape = value[0:2]

            # Regenerate the camera calibration matrix
            _, self.camera_matrix, self.dist_coeffs, _, _ = \
                cv2.calibrateCamera(self.obj_points, self.img_points,
                                    self.shape, None, None)

    def process(self, output):
        """Process the input video and dump it into the output.

        :param output: string
            File name of the output video.
        """
        processed_clip = self.clip.fl_image(self._process_image)
        processed_clip.to_videofile(output, audio=False)

    def process_video_image(self, iframe=1):
        """Process a frame in a video

        :param iframe: int
            Frame index.

        :return: processed image.
        """
        self.iframe = iframe
        img = self.clip.get_frame(int(iframe)/self.clip.fps)

        return self._process_image(img)

    def get_video_image(self, iframe=1):
        """get a frame in a video

        :param iframe: int
            Frame index.

        :return: frame image.
        """
        self.iframe = iframe
        img = self.clip.get_frame(int(iframe)/self.clip.fps)

        return img

    def _process_image(self, img):
        """Process an image.

        :param img: numpy.ndarray
            Original image.

        :return: processed: numpy.ndarray
            Processed image.
        """
        self.shape = img.shape

        undistorted = cv2.undistort(
            img, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)

        if self._is_search_laneline is True:
            processed = self._search_lanelines(undistorted)

            processed = self._draw_center_indicator(processed)
            processed = self._draw_text(processed)
        else:
            processed = np.copy(undistorted)

        if self._is_search_car is True:
            boxes = search_cars(processed, self.car_classifier,
                                scale_ratios=(0.5, 0.7), confidence_thresh=0.2,
                                overlap_thresh=0.2, step_size=(0.125, 0.125),
                                region=((0, 0.5), (1.0, 0.9)))
            processed = draw_boxes(processed, boxes)

        return processed

    def _search_lanelines(self, img):
        """Search and draw lane lines in an image

        :param img: numpy.ndarray
            Original image.

        :return img_with_lanelines: numpy.ndarray
            Image with lane line drawn in it.
        """
        # Applying threshold
        threshed = self.thresh(img)

        # Transform to bird-eye view
        bird_eye = cv2.warpPerspective(
            threshed, self.perspective_matrix, threshed.shape[:2][::-1])

        # Search line in the warped image
        if self.lines is None:
            self.lines = TwinLine(np.arange(bird_eye.shape[0]))
            self.lines.left.max_fail = self.clip.fps*self.max_poor_fit_time
            self.lines.left.max_fail = self.clip.fps*self.max_poor_fit_time

        self.lines.search(bird_eye)

        # Draw lines in the original image
        warp_zero = np.zeros_like(bird_eye).astype(np.uint8)
        colored_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        if self.lines.left.ave_x.any() and self.lines.right.ave_x.any():

            pts = np.hstack(((self.lines.left.ave_x, self.lines.left.y),
                            (self.lines.right.ave_x[::-1], self.lines.right.y[::-1]))).T

            # Draw a green polygon in a black background
            cv2.fillPoly(colored_warp, np.int_([pts]), (0, 255, 0))

            # Draw the left line
            left_pts = np.array((self.lines.left.ave_x, self.lines.left.y)).T
            cv2.polylines(colored_warp, np.int32([left_pts]), 0, (255, 255, 0), thickness=30)

            # Draw the right line
            right_pts = np.array((self.lines.right.ave_x, self.lines.right.y)).T
            cv2.polylines(colored_warp, np.int32([right_pts]), 0, (255, 255, 0), thickness=30)

            # Draw a line corresponding to the center of the two lane lines
            center_pts = (left_pts[-50:] + right_pts[-50:])/2
            cv2.polylines(colored_warp, np.int32([center_pts]), 0, (255, 255, 0), thickness=5)

        elif self.lines.left.ave_x.any():
            # Draw the left line only
            left_pts = np.array((self.lines.left.ave_x, self.lines.left.y)).T
            cv2.polylines(colored_warp, np.int32([left_pts]), 0, (255, 255, 0), thickness=30)

        elif self.lines.right.ave_x.any():
            # Draw the right line only
            right_pts = np.array((self.lines.right.ave_x, self.lines.right.y)).T
            cv2.polylines(colored_warp, np.int32([right_pts]), 0, (255, 255, 0), thickness=30)

        # Applying inverse perspective transform
        inv_warp = cv2.warpPerspective(
            colored_warp, self.inv_perspective_matrix, colored_warp.shape[:2][::-1])

        # Draw lines in the original image
        img_with_lanelines = cv2.addWeighted(img, 1.0, inv_warp, 0.5, 0.0)

        return img_with_lanelines

    def thresh(self, img):
        """Apply the combination of different thresholds

        :param img: numpy.ndarray
            Original image.

        :return threshed: numpy.ndarray
            Image after applying threshold.
        """
        # Apply gradient and color threshold
        binary = None
        for param in self.thresh_params:
            th = Threshold(img, param['color_space'], param['channel'])

            th.transform(param['direction'], thresh=param['thresh'])
            if binary is None:
                binary = th.binary
            else:
                binary |= th.binary

        # Remove the influence from the front of the car
        binary[-40:, :] = 0

        return binary

    def _draw_center_indicator(self, img):
        """Draw two lines

        One refers to the center of the car, and the other refers to
        the center of the two lane lines.

        :param img: numpy.ndarray
            Original image.

        :return new_img: numpy.ndarray.
            Processed image.
        """
        new_img = np.copy(img)

        # Assume the camera is at the center of the car
        car_center_pts = np.vstack([np.ones(50)*new_img.shape[1]/2,
                                    np.arange(new_img.shape[0])[-50:]]).T
        cv2.polylines(new_img, np.int32([car_center_pts]), 0, (0, 0, 0), thickness=5)

        # Draw the center of the two lane lines
        if self.lines.left.ave_x.any() and self.lines.right.ave_x.any():
            off_center = (self.lines.left_space - self.lines.right_space)/2.0
            cv2.putText(new_img, "off center: {:.1} m".format(off_center),
                        (int(new_img.shape[1]/2 - 120), new_img.shape[0] - 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return new_img

    def _draw_text(self, img):
        """Put texts on an image.

        :param img: numpy.ndarray
            Original image.

        :return new_img: numpy.ndarray.
            Processed image.
        """
        new_img = np.copy(img)

        c_norm = (255, 255, 255)
        c_warning = (180, 0, 0)
        font_name = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.0
        font_thickness = 2

        x_text = 40  # Start x position
        y_text = 40  # Start y position
        y_text_space = 45

        # Frame count
        text_string = "Frame: {}".format(self.iframe)
        cv2.putText(new_img, text_string, (x_text, y_text),
                    font_name, font_scale, (255, 50, 200), font_thickness)

        # Line information
        left_space = {'text': '', 'color': c_norm}
        right_space = {'text': '', 'color': c_norm}

        if self.lines.left.ave_p.any() and self.lines.right.ave_p.any():
            left_space['text'] = "To left laneline: {:.1f} m".\
                format(self.lines.left_space)

            right_space['text'] = "To right laneline: {:.1f} m".\
                format(self.lines.right_space)

        elif self.lines.left.ave_p.any():
            left_space['text'] = "To left laneline: {:.1f} m".\
                format(self.lines.left_space)

            right_space['text'] = "To right laneline: unknown!"
            right_space['color'] = c_warning
        elif self.lines.right.ave_p.any():
            left_space['text'] = "To left laneline: unknown!"
            left_space['color'] = c_warning

            right_space['text'] = "To right laneline: {:.1f} m".\
                format(self.lines.right_space)
        else:
            left_space['text'] = "To left laneline: unknown!"
            left_space['color'] = c_warning
            right_space['text'] = "To right laneline: unknown!"
            right_space['color'] = c_warning

        y_text += y_text_space
        cv2.putText(new_img, left_space['text'], (x_text, y_text),
                    font_name, font_scale, left_space['color'],
                    font_thickness)
        y_text += y_text_space
        cv2.putText(new_img, right_space['text'], (x_text, y_text),
                    font_name, font_scale, right_space['color'],
                    font_thickness)

        # Bending information
        local_bending_radius = {'text': '', 'color': c_norm}
        ahead_bending_angle = {'text': '', 'color': c_norm}
        # Radius of curvature
        if self.lines.local_bend_radius is None:
            local_bending_radius['text'] = "Local bending radius: Unknown"
            local_bending_radius['color'] = c_warning
            ahead_bending_angle['text'] = "Ahead bending angle: Unknown"
            ahead_bending_angle['color'] = c_warning
        else:
            local_bending_radius['text'] = \
                "Local bending radius: {:.0f} m".format(
                    self.lines.local_bend_radius)

            ahead_bending_angle['text'] = \
                "Ahead bending angle: {:.1f} deg".format(
                    self.lines.ahead_bend_angle)

        y_text += y_text_space
        cv2.putText(new_img, local_bending_radius['text'], (x_text, y_text),
                    font_name, font_scale, local_bending_radius['color'],
                    font_thickness)
        y_text += y_text_space
        cv2.putText(new_img, ahead_bending_angle['text'], (x_text, y_text),
                    font_name, font_scale, ahead_bending_angle['color'],
                    font_thickness)

        return new_img