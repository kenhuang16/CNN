#!/usr/bin/python
"""
class:
    - TrafficVideo()
"""
import pickle
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

from lane_line import LaneLine, sample_lines
from utilities import draw_box, non_maxima_suppression, get_perspective_trans_matrix
from threshold import thresh_image
from parameters import INF, Y_METER_PER_PIXEL, X_METER_PER_PIXEL


class TrafficVideo(object):
    """TrafficVideo class"""
    def __init__(self, video_clip, is_search_laneline=True, is_search_car=True,
                 camera_cali_file='', perspective_trans_params=None,
                 thresh_params=None,
                 car_classifier_file='', car_search_params=None):
        """Initialization.

        :param video_clip: string
            File name of the input video.
        :param camera_cali_file: string
            Name of the pickle file storing the camera calibration
        :param perspective_trans_params: tuple
            Source and destination points for perspective transformation
            For example, (src, dst) with
                src = np.float32([[0, 720], [570, 450], [708, 450], [1280, 720]])
                dst = np.float32([[0, 720], [0, 0], [1280, 0], [1280, 720]])
        :param thresh_params: list of dictionary
            Parameters for the gradient and color threshhold.
        :param car_classifier_file: String
            Pickle file for CarClassifier object.
        :param car_search_params: dictionary
            Parameters for performing car search in an image.
        :param is_search_car: Boolean
            True for search cars in the video/image.
        :param is_search_laneline: Boolean
            True for search lanelines in the video/image.
        """
        self.clip = VideoFileClip(video_clip)
        self.shape = self.get_video_image(0).shape
        assert(len(self.shape) == 3)  # colored image
        self.frame = 0  # the current frame index

        if is_search_car is False and is_search_laneline is False:
            raise SystemExit("Nothing needs to be done!:(")

        # Load camera calibration information
        with open(camera_cali_file, "rb") as fp:
            camera_cali = pickle.load(fp)
        # Get required parameters for image undistortion
        _, self.camera_matrix, self.dist_coeffs, _, _ = cv2.calibrateCamera(
            camera_cali["obj_points"], camera_cali["img_points"],
            self.shape[0:2], None, None)

        self._is_search_laneline = is_search_laneline
        self.thresh_params = thresh_params  # threshold parameter in lane line search
        self.ppt_trans_params = perspective_trans_params
        # perspective and inverse perspective transformation matrices
        self.ppt_trans_matrix, self.inv_ppt_trans_matrix = \
            get_perspective_trans_matrix(self.ppt_trans_params[0],
                                         self.ppt_trans_params[1])
        y_fit = np.arange(self.shape[0], 0, -10)
        self.left_line = LaneLine(y_fit)
        self.right_line = LaneLine(y_fit)

        self._is_search_car = is_search_car
        with open(car_classifier_file, "rb") as fp:
            self.car_classifier = pickle.load(fp)
        self.car_search_params = car_search_params
        self.car_boxes = []  # square boxes classified as cars

    def process(self, output):
        """Process the input video and dump it into the output.

        :param output: string
            File name of the output video.
        """
        processed_clip = self.clip.fl_image(self._process_image)
        processed_clip.to_videofile(output, audio=False)

    def process_video_image(self, frame=0):
        """Process a frame in a video

        :param frame: int
            Frame index.

        :return: processed image.
        """
        self.frame = frame
        img = self.clip.get_frame(int(frame)/self.clip.fps)

        return self._process_image(img)

    def get_video_image(self, frame=0):
        """get a frame in a video

        :param frame: int
            Frame index.

        :return: frame image.
        """
        img = self.clip.get_frame(int(frame)/self.clip.fps)

        return img

    def _process_image(self, img):
        """Process an image.

        :param img: numpy.ndarray
            Original image.

        :return: processed: numpy.ndarray
            Processed image.
        """
        undistorted = cv2.undistort(
            img, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)

        processed = np.copy(undistorted)
        if self._is_search_laneline is True:
            warped = cv2.warpPerspective(
                img, self.ppt_trans_matrix, img.shape[:2][::-1])
            warped = cv2.blur(warped, (15, 5), 0)
            threshed = thresh_image(warped, self.thresh_params)

            points = sample_lines(threshed)
            left = []
            right = []
            for i in range(len(points)):
                if points[i][0][0] < warped.shape[1] / 2:
                    left.append(i)
                elif points[i][0][0] > warped.shape[1] / 2:
                    right.append(i)

            if len(left) > 0:
                self.left_line.points = points[left[0]]
            else:
                self.left_line.points = None

            if len(right) > 0:
                self.right_line.points = points[right[-1]]
            else:
                self.right_line.points = None

            # Draw blue lane lines in a black ground
            bkg = np.zeros_like(processed).astype(np.uint8)
            for line in (self.left_line, self.right_line):
                if line.x_fit is not None:
                    cv2.polylines(
                        bkg, np.int32([[np.vstack((line.x_fit, line.y_fit)).T]]),
                        0, (0, 0, 255), thickness=30)

            # Draw a green polygon in a black background
            if self.left_line.p_fit is not None and self.right_line.p_fit is not None:
                pts = np.hstack(((self.left_line.x_fit,
                                  self.left_line.y_fit),
                                 (self.right_line.x_fit[::-1],
                                  self.right_line.y_fit[::-1]))).T
                cv2.fillPoly(bkg, np.int_([pts]), (0, 255, 0))

            # Inverse perspective transformation
            lines = cv2.warpPerspective(bkg, self.inv_ppt_trans_matrix,
                                        bkg.shape[:2][::-1])
            # draw the lines on the image
            processed = cv2.addWeighted(processed, 1.0, lines, 0.5, 0.0)

        if self._is_search_car is True:
            boxes = sw_search_car(
                undistorted, self.car_classifier,
                scales=self.car_search_params['scales'],
                confidence_thresh=self.car_search_params['confidence_thresh'],
                overlap_thresh=self.car_search_params['overlap_thresh'],
                step_size=self.car_search_params['step_size'],
                regions=self.car_search_params['regions'])

            # draw the boxes on the image
            processed = draw_box(processed, boxes)

        processed = self._draw_text(processed)

        self.frame += 1

        return processed

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
        text_string = "Frame: {}".format(self.frame)
        cv2.putText(new_img, text_string, (x_text, y_text),
                    font_name, font_scale, (255, 50, 200), font_thickness)

        # Line information
        left_space = {'text': '', 'color': c_norm}
        right_space = {'text': '', 'color': c_norm}

        if self.left_line.p_fit is not None and self.right_line.p_fit is not None:
            left_space['text'] = "To left laneline: {:.1f} m".\
                format(X_METER_PER_PIXEL*(self.shape[1]/2 - self.left_line.x_fit[0]))
            right_space['text'] = "To right laneline: {:.1f} m".\
                format(X_METER_PER_PIXEL*(self.right_line.x_fit[0] - self.shape[1]/2))

        elif self.left_line.p_fit is not None:
            left_space['text'] = "To left laneline: {:.1f} m".\
                format(X_METER_PER_PIXEL*(self.shape[1]/2 - self.left_line.x_fit[0]))

            right_space['text'] = "To right laneline: unknown!"
            right_space['color'] = c_warning

        elif self.right_line.p_fit is not None:
            left_space['text'] = "To left laneline: unknown!"
            left_space['color'] = c_warning

            right_space['text'] = "To right laneline: {:.1f} m".\
                format(X_METER_PER_PIXEL*(self.right_line.x_fit[0] - self.shape[1]/2))
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
        # if self.lines.local_bend_radius is None:
        #     local_bending_radius['text'] = "Local bending radius: Unknown"
        #     local_bending_radius['color'] = c_warning
        #     ahead_bending_angle['text'] = "Ahead bending angle: Unknown"
        #     ahead_bending_angle['color'] = c_warning
        # else:
        #     local_bending_radius['text'] = \
        #         "Local bending radius: {:.0f} m".format(
        #             self.lines.local_bend_radius)
        #
        #     ahead_bending_angle['text'] = \
        #         "Ahead bending angle: {:.1f} deg".format(
        #             self.lines.ahead_bend_angle)
        # @staticmethod
        # def _bend_radius(p, y):
        #     """"""
        #     A = p[0] / Y_METER_PER_PIXEL ** 2 * X_METER_PER_PIXEL
        #     B = p[1] / Y_METER_PER_PIXEL * X_METER_PER_PIXEL
        #
        #     return (1 + (2 * A * y[-1] + B) ** 2) ** 1.5 / (2 * A)
        #
        # @staticmethod
        # def _bend_angle(x, y):
        #     """"""
        #     bend_angle = np.arctan((x[0] - x[2]) / (y[2] - y[0])
        #                            * X_METER_PER_PIXEL / Y_METER_PER_PIXEL)
        #
        #     return bend_angle * 180 / np.pi

        y_text += y_text_space
        cv2.putText(new_img, local_bending_radius['text'], (x_text, y_text),
                    font_name, font_scale, local_bending_radius['color'],
                    font_thickness)
        y_text += y_text_space
        cv2.putText(new_img, ahead_bending_angle['text'], (x_text, y_text),
                    font_name, font_scale, ahead_bending_angle['color'],
                    font_thickness)

        return new_img


def sw_search_car(img, classifier, step_size=(1.0, 1.0),
                  scales=(1.0,), regions=(((0.0, 0.0), (1.0, 1.0)),),
                  confidence_thresh=0.0, overlap_thresh=0.5, heat_thresh=0.5):
    """Search cars in an image

    :param img: numpy.ndarray
        Original image.
    :param classifier: CarClassifier object
        Classifier.
    :param step_size: 1x2 tuple, float
        Size of the sliding step in the unit of the image shape.
    :param scales: a tuple of float
        Scales of the original image.
    :param regions: a tuple of ((x0, y0), (x1, y1))
        Search region of the image in fraction.
    :param confidence_thresh: float
        Threshold of the classification score.
    :param overlap_thresh: float, between 0 and 1
        Overlap threshold when applying non-maxima-suppression.
    :param heat_thresh: float
        Threshold of the heat map.

    :return car_boxes: list of ((x0, y0), (x1, y1))
        Diagonal coordinates of the predicted boxes.
    """
    # Applying sliding window search with different scale ratios
    window_coordinates = []
    window_confidences = []
    for scale, region in zip(scales, regions):
        x0 = int(region[0][0] * img.shape[1])
        x1 = int(region[1][0] * img.shape[1])
        y0 = int(region[0][1] * img.shape[0])
        y1 = int(region[1][1] * img.shape[0])
        search_region = img[y0:y1, x0:x1, :]

        scores, windows = classifier.sliding_window_predict(
            search_region, step_size=step_size, binary=False, scale=scale)

        for window, score in zip(windows, scores):
            if score > confidence_thresh:
                point1 = (window[0][0] + x0, window[0][1] + y0)
                point2 = (window[1][0] + x0, window[1][1] + y0)
                window_coordinates.append((point1, point2))
                window_confidences.append(score)

    # Apply the 'non_maxima_suppression' to filter the windows
    car_boxes = non_maxima_suppression(
        window_coordinates, window_confidences, overlap_thresh)

    # Merge the overlapped boxes
    # Note: merge the connected box is not a good idea, especially when
    # several cars overlap!
    # car_boxes = merge_box(car_boxes, img.shape[:2])

    # select the box by 'heat' (sum of confidence)
    # Note: select by heat results in a box which is much bigger than
    # the car!
    # car_boxes = box_by_heat(windows_coordinates_threshed,
    #                         windows_confidences_threshed,
    #                         img.shape[:2], thresh=heat_thresh)

    return car_boxes
