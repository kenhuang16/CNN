#!/usr/bin/python
"""
"""
import time
import pickle
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label, center_of_mass

from lane_line import TwinLine
from my_plot import double_plot
from threshold import Threshold
from calibration import undistort_image
from car import Car


INF = 1.0e21
# The length (in pixel) of the front of the car in the original image.
CAR_FRONT_LENGTH = 40

DEBUG = False
DEBUG_LANELINE = False
TIME_MONITOR = False

SEARCH_LANELINE = False
SEARCH_CAR = True


class TrafficVideo(object):
    """TrafficVideo class

    Attributes
    ----------


    """
    def __init__(self, input, camera_cali_file=None, perspective_trans_file=None,
                 thresh_params=None, max_poor_fit_time=0.5, car_classifier=None):
        """Initialization.

        Parameters
        ----------
        input: string
            File name of the input video.
        camera_cali_file: string
            Name of the pickle file storing the camera calibration
        perspective_trans_file: string
            Name of the pickle file storing the perspective transform
            matrices.
        thresh_params: list of dictionary
            Parameters for the gradient and color threshhold.
        max_poor_fit_time: float
            Maximum allowed period (in second) of consecutive fail before
            a fresh line search.
        car_classifier:
            Car classifier pickle file.
        """
        self.input = input
        self.clip = VideoFileClip(input)
        self.frame = 0
        self._shape = None

        self.lines = None
        self._cars = {}

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

        self.search_lanelines = True
        self.search_cars = True

        try:
            with open(car_classifier, "rb") as fp:
                self.car_cls = pickle.load(fp)
        except IOError:
            print("Warning: car classifier is missing!")

        self.direction_text_string = ''

        self.car_heatmap = []
        self.n_heatmap_sum = 5
        self.car_heatmap_thresh = 50
        self.car_min_width = 32
        self.car_min_height = 32

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

        Parameters
        ----------
        output: string
            File name of the output video.
        """
        processed_clip = self.clip.fl_image(self._process_image)
        processed_clip.to_videofile(output, audio=False)

    def process_video_image(self, frame=1):
        """Process a frame in a video

        Parameters
        ----------
        frame: int
            Frame index.

        Return
        ------
        Processed image array.
        """
        self.frame = frame
        img = self.clip.get_frame(int(frame)/self.clip.fps)

        return self._process_image(img)

    def _process_image(self, img):
        """Process an image.

        Parameters
        ----------
        img: numpy.ndarray
            Image array.

        Return
        ------
        processed_image: numpy.ndarray
            Processed image array.
        """
        self.shape = img.shape

        undistorted = cv2.undistort(
            img, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)

        if SEARCH_LANELINE is True:
            t0 = time.time()

            processed = self._search_lanelines(undistorted)
            if TIME_MONITOR is True:
                print("TOTAL: found lane ines in {:.2} s\n".format(time.time() - t0))

            processed = self._draw_center_indicator(processed)
            processed = self._draw_text(processed)
        else:
            processed = np.copy(undistorted)

        if SEARCH_CAR is True:
            t0 = time.time()

            car_windows = self._search_cars(undistorted)
            processed = self._draw_windows(processed, car_windows)

            if TIME_MONITOR is True:
                print("TOTAL: Found cars in {:.2} s\n".format(time.time() - t0))

        return processed

    def _search_lanelines(self, img):
        """Search and draw lane lines in an image

        Parameters
        ----------
        img: numpy.ndarray
            Image array.

        Return
        ------
        img_with_lanelines: numpy.ndarray
            Image array with lane line drawn in it.
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

        Parameters
        ----------
        img: numpy.ndarray
            Image array.

        Return
        ------
        Image array after applying threshold.
        """
        # Apply gradient and color threshold
        th = Threshold(img)
        for param in self.thresh_params:
            th.transform(param['channel'], param['direct'], thresh=param['thresh'])

        threshed = th.binary

        # Remove the influence from the front of the car
        threshed[-40:, :] = 0

        return threshed

    def _draw_center_indicator(self, img):
        """Draw two lines

        One refers to the center of the car, and the other refers to
        the center of the two lanelines.

        Paramter:
        ----------
        img: numpy.ndarray
            Image array.

        Return:
        -------
        Image numpy.ndarray.
        """
        # Assume the camera is at the center of the car
        car_center_pts = np.vstack([np.ones(50)*img.shape[1]/2,
                                    np.arange(img.shape[0])[-50:]]).T
        cv2.polylines(img, np.int32([car_center_pts]), 0, (0, 0, 0), thickness=5)

        # Draw the center of the two lanelines
        if self.lines.left.ave_x.any() and self.lines.right.ave_x.any():
            off_center = (self.lines.left_space - self.lines.right_space)/2.0
            cv2.putText(img, "off center: {:.1} m".format(off_center),
                        (int(img.shape[1]/2 - 120), img.shape[0] - 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return img

    def _draw_text(self, img):
        """Put texts on an image.

        Parameters
        ----------
        img: numpy.ndarray
            Image array.
        """
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
        cv2.putText(img, text_string, (x_text, y_text),
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
        cv2.putText(img, left_space['text'], (x_text, y_text),
                    font_name, font_scale, left_space['color'],
                    font_thickness)
        y_text += y_text_space
        cv2.putText(img, right_space['text'], (x_text, y_text),
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
        cv2.putText(img, local_bending_radius['text'], (x_text, y_text),
                    font_name, font_scale, local_bending_radius['color'],
                    font_thickness)
        y_text += y_text_space
        cv2.putText(img, ahead_bending_angle['text'], (x_text, y_text),
                    font_name, font_scale, ahead_bending_angle['color'],
                    font_thickness)

        return img

    def _search_cars(self, img):
        """Search cars in an image

        Parameters
        ----------
        img: numpy.ndarray
            Image array.

        Return
        ------
        cars: list of ((x0, y0), (x1, y1))
            Diagonal coordinates of the windows containing cars.
        """
        t0 = time.time()

        window_size = 64
        window_size_increase = 64
        x_slide_step = 0.125
        y_slide_step = 0.125
        y_window_extra = y_slide_step*8
        y0_shift = 40
        x0_shift = 80

        x0 = 300
        x1 = 1280
        y0 = np.int(img.shape[0]/2 - window_size*y_slide_step*2) + y0_shift
        y1 = y0 + np.int(window_size*(1 + y_window_extra))

        search_regions = []
        tp_windows = []
        dt_feature_extraction = 0.0
        dt_prediction = 0.0
        while window_size <= 196:
            search_regions.append(((x0, y0), (x1, y1)))
            x_slide_step_size = np.int(x_slide_step * window_size)
            y_slide_step_size = np.int(y_slide_step * window_size)

            # print("window_size: {}, slide_step_size: {}"
            #       .format(window_size, slide_step_size))
            # print("y0: {}, y1: {}".format(y0, y1))
            t0 = time.time()
            features, windows = self.car_cls.extract(
                img, window_shape=(window_size, window_size),
                slide_step=(x_slide_step_size, y_slide_step_size),
                x_range=(x0, x1), y_range=(y0, y1), x_pad=window_size)

            t1 = time.time()
            dt_feature_extraction += t1 - t0

            predictions = self.car_cls.predict(features)

            dt_prediction += time.time() - t1

            for prediction, window in zip(predictions, windows):
                if prediction == 1:
                    tp_windows.append(window)

            window_size += window_size_increase

            x0 -= x0_shift
            if x0 < 0:
                x0 = 0
            x1 += x0_shift
            if x1 > img.shape[1]:
                x1 = img.shape[1]
            y0 = np.int(img.shape[0]/2 - window_size*y_slide_step*2) + y0_shift
            if y0 > img.shape[0]:
                y0 = img.shape[0]
            y1 = y0 + np.int(window_size*(1 + y_window_extra))
            if y1 > img.shape[0] - CAR_FRONT_LENGTH:
                y1 = img.shape[0] - CAR_FRONT_LENGTH

        # if DEBUG is True:
        #     _, ax = plt.subplots(figsize=(16, 9))
        #     img_with_search_regions = self._draw_windows(img, search_regions)
        #     ax.imshow(img_with_search_regions)
        #     plt.tight_layout()
        #     plt.show()

        if TIME_MONITOR is True:
            print("- Feature extractions in {:.2} s".format(dt_feature_extraction))
            print("- Prediction in {:.2} s".format(dt_prediction))

        # Create heatmap
        heatmap = np.zeros_like(img[:, :, 0])
        for window in tp_windows:
            heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1

        # Sum heatmap over history and apply threshold
        self.car_heatmap.append(heatmap)
        if len(self.car_heatmap) > self.n_heatmap_sum:
            self.car_heatmap.pop(0)
        sum_heatmap = sum(self.car_heatmap)
        sum_heatmap[sum_heatmap < self.car_heatmap_thresh] = 0

        # Labeling heatmap
        t0 = time.time()
        labels = label(sum_heatmap)

        if DEBUG is True:
            self._visualize_search_result(img, tp_windows, sum_heatmap, labels)

        if TIME_MONITOR is True:
            print("- Labeling in {:.2} s".format(time.time() - t0))

        car_windows = []
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y

            heat_region = heatmap[nonzeroy.min():nonzeroy.max(),
                          nonzerox.min():nonzerox.max()]

            if np.max(nonzerox) - np.min(nonzerox) < self.car_min_width or \
                np.max(nonzeroy) - np.min(nonzeroy) < self.car_min_height:
                # Reject if the region is too small
                continue

            car_window = ((np.min(nonzerox), np.min(nonzeroy)),
                              (np.max(nonzerox), np.max(nonzeroy)))
            car_windows.append(car_window)

        return car_windows

    @staticmethod
    def _visualize_search_result(img, windows, heatmap, labels):
        """Visualize windows and the corresponding heat maps

        Parameters
        ----------
        img: numpy.ndarray
            Image array.
        windows: list of ((x0, y0), (x1, y1))
            Diagonal coordinates of the windows.
        heatmap: numpy.ndarray
            Heatmap Image array.
        labels: numpy.ndarray
            Result from skimage.label().

        For debug only.
        """
        img_with_windows = np.copy(img)

        for window in windows:
            cv2.rectangle(img_with_windows, window[0], window[1], (0, 0, 255), 6)

        fig, ax = plt.subplots(3, 1, figsize=(6, 13.5))

        ax[0].imshow(img_with_windows)
        ax[0].set_title("raw search result", fontsize=20)
        ax[1].imshow(heatmap, cmap='hot')
        ax[1].set_title("heatmap", fontsize=20)
        ax[2].imshow(labels[0], cmap='gray')
        ax[2].set_title("labeled", fontsize=20)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _draw_windows(img, windows, color=(0, 0, 255), thickness=2):
        """Draw rectangular windows in an image

        Parameters
        ----------
        img: numpy.ndarray
            Image array.
        windows: list of ((x0, y0), (x1, y1))
            Diagonal coordinates of the windows.
        color: tuple
            RGB color tuple.
        thickness: int
            Line thickness.

        Return
        ------
        New image array with windows imprinted.
        """
        new_img = np.copy(img)
        for window in windows:
            cv2.rectangle(new_img, window[0], window[1], color, thickness)
        # Return the image
        return new_img

    def set_car(self, *args, **kwargs):
        """Add a car object"""
        i = self.first_available_index(self._cars)

        if (len(args) > 0) and isinstance(args[0], Car):
            self._cars[i] = args[0]
        else:
            try:
                self._cars[i] = Car(*args, **kwargs)
            except:
                raise ValueError(
                    "Input is not valid for a Car object instance!\n")

    def get_car(self, i):
        """Get a car object by index

        Parameters
        ----------
        i: int
            Car index.
        """
        if not isinstance(i, int) or i < 0:
            raise ValueError("i must be a non-negative integer!")

        return self._cars[i]

    def get_carset(self):
        """Get the set of cars"""
        return self._cars

    def del_car(self, i):
        """Delete a car object

        Parameters
        ----------
        i: int
            Car index.
        """
        if i not in self._cars:
            raise ValueError("Index is not found.\n")
        else:
            del self._cars[i]

    @staticmethod
    def first_available_index(set):
        """List First Unused Index from Variable Objects List

        Parameters
        ----------
        set:
        """
        i = 0
        while i in set:
            i += 1

        return i
