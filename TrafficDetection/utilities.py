"""
- function change_colorspace()

Convert the color space of an image

- function non_maxima_suppression()

Apply the non-maxima suppresion to boxes.

- function draw_windows()

Draw rectangular windows in an image.

- function two_plots()

Plot the original and processed image together.

"""
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt


def change_colorspace(img, color_space):
    """Convert the color space of an image

    The readin image is assumed to have RGB color space.

    :param img: numpy.ndarray
        Original image.
    :param color_space: string
        Color space of the new image.

    :return new_img: numpy.ndarray
        Image after changing color space
    """
    if len(img.shape) < 3:
        raise ValueError("A color image is required!")

    if re.search(r'^GRAY', color_space, re.IGNORECASE):
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif re.search(r'^HSV', color_space, re.IGNORECASE):
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif re.search(r'^LUV', color_space, re.IGNORECASE):
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    elif re.search(r'^HLS', color_space, re.IGNORECASE):
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif re.search(r'^YUV', color_space, re.IGNORECASE):
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    elif re.search(r'^YCrCb', color_space, re.IGNORECASE):
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif color_space == 'RGB':
        new_img = np.copy(img)
    else:
        raise ValueError("Unknown color space!")

    return new_img


def non_maxima_suppression(boxes, scores, threshold=0.5):
    """Apply the non-maxima suppresion to boxes

    For boxes having overlap higher than the threshold, only the box
    with the highest score will be kept.

    :param boxes: list of ((x0, y0), (x1, y1))
        Diagonal coordinates of the boxes.
    :param scores: list
        Scores of the boxes.
    :param threshold: float, between 0 and 1
        Overlap threshold.

    :return a list of diagonal coordinates of the selected boxes.
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    assert len(boxes) == len(scores)

    # reformat the coordinates of the bounding boxes
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for box in boxes:
        x1.append(box[0][0])
        y1.append(box[0][1])
        x2.append(box[1][0])
        y2.append(box[1][1])
    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)

    # compute the area of all bounding boxes
    area = (x2 - x1 + 1)*(y2 - y1 + 1)

    # sort the index by scores (ascending)
    sorted_index = np.argsort(scores)

    pick = []   # the list of picked indexes
    while len(sorted_index) > 0:
        # add the box with the highest score
        i = sorted_index[-1]
        pick.append(boxes[i])

        # In the following, boxes overlapping with the box having the
        # highest score will be removed together with the box having the
        # highest score from sorted_index.

        # ------
        # np.maximum(a, b), a is a number, b is a list:
        # Any element in b will be replaced by a if it is less than a.
        # ------

        xx1 = np.maximum(x1[i], x1[sorted_index[:-1]])
        yy1 = np.maximum(y1[i], y1[sorted_index[:-1]])
        xx2 = np.minimum(x2[i], x2[sorted_index[:-1]])
        yy2 = np.minimum(y2[i], y2[sorted_index[:-1]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[sorted_index[:-1]]

        # delete all indexes from the index list that have
        sorted_index = np.delete(
            sorted_index, np.concatenate([[len(sorted_index) - 1],
                                          np.where(overlap > threshold)[0]]))

    return pick


def sliding_window(img, window_size=(64, 64), step_size=(16, 16), scale=(1.0, 1.0)):
    """Search cars in an image

    :param img: numpy.ndarray
        Image array.
    :param window_size: 1x2 tuple, int
        Size of the sliding window.
    :param step_size: 1x2 tuple, int
        Size of the sliding step.
    :param scale: 1x2 tuple, float
        Scale of the original image

    :return windows: list
        A list of window images with the size of "window_size"
    """
    new_x_size = np.int(img.shape[1]*scale[1])
    new_y_size = np.int(img.shape[0]*scale[0])
    img_resized = cv2.resize(img, (new_x_size, new_y_size))

    # plt.imshow(img_resized)
    # plt.show()

    windows = []
    y0_window = 0
    while (y0_window + window_size[0]) <= img_resized.shape[0]:
        x0_window = 0
        while (x0_window + window_size[0]) <= img_resized.shape[1]:

            windows.append(img_resized[y0_window:(y0_window + window_size[0]),
                                       x0_window:(x0_window + window_size[1])])

            x0_window += step_size[1]

        y0_window += step_size[0]

    return windows


def draw_windows(img, windows, color=(0, 0, 255), thickness=4):
    """Draw rectangular windows in an image

    :param img: numpy.ndarray
        Original image.
    :param windows: list of ((x0, y0), (x1, y1))
        Diagonal coordinates of the windows.
    :param color: tuple
        RGB color tuple.
    :param thickness: int
        Line thickness.

    :return new_img: numpy.ndarray
        New image with windows imprinted
    """
    new_img = np.copy(img)
    for window in windows:
        cv2.rectangle(new_img, window[0], window[1], color, thickness)

    return new_img


    # Create a black background image
    # heatmap = np.zeros(img.shape[0:2])
    #
    # for window, confidence in zip(windows_coordinates, windows_confidences):
    #     if confidence > 0.5:
    #         heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += confidence

    # Labeling heatmap
    # labels = label(heatmap)
    #
    # self._visualize_search_result(img, windows_coordinates, heatmap, labels)

    # heatmap = []
    # for car_number in range(1, labels[1] + 1):
    #     # Find pixels with each car_number label value
    #     nonzero = (labels[0] == car_number).nonzero()
    #     # Identify x and y values of those pixels
    #     nonzeroy = np.array(nonzero[0])
    #     nonzerox = np.array(nonzero[1])
    #
    #     x_span = np.max(nonzerox) - np.min(nonzerox)
    #     y_span = np.max(nonzeroy) - np.min(nonzeroy)
    #     if x_span >= self._car_minimum_size and y_span >= self._car_minimum_size:
    #         heatmap.append(((np.min(nonzerox), np.min(nonzeroy)),
    #                        (np.max(nonzerox), np.max(nonzeroy))))

    # @staticmethod
    # def _visualize_search_result(img, windows, heatmap, labels):
    #     """Visualize windows and the corresponding heat maps
    #
    #     Parameters
    #     ----------
    #     img: numpy.ndarray
    #         Image array.
    #     windows: list of ((x0, y0), (x1, y1))
    #         Diagonal coordinates of the windows.
    #     heatmap: numpy.ndarray
    #         Heatmap Image array.
    #     labels: numpy.ndarray
    #         Result from skimage.label().
    #
    #     For debug only.
    #     """
    #     img_with_windows = np.copy(img)
    #
    #     for window in windows:
    #         cv2.rectangle(img_with_windows, window[0], window[1], (0, 0, 255), 6)
    #
    #     fig, ax = plt.subplots(3, 1, figsize=(4.5, 12))
    #
    #     ax[0].imshow(img_with_windows)
    #     ax[0].set_title("raw search result", fontsize=18)
    #     ax[1].imshow(heatmap, cmap='hot')
    #     ax[1].set_title("heatmap", fontsize=18)
    #     ax[2].imshow(labels[0], cmap='gray')
    #     ax[2].set_title("labeled", fontsize=18)
    #
    #     plt.tight_layout()
    #     plt.show()


def two_plots(img1, img2, titles=('', '', ''), output=''):
    """Plot two images together

    It is used to draw a comparison between the original and
    final images.

    :param img1: numpy.ndarray
        The first image.
    :param img2: numpy.ndarray
        The second image.
    :param titles: tuple
        Titles in the form ('first', 'second', 'super title')
    :param output: string
        Name of the output image if specified.
    """
    ft_size = 18

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    if len(img1.shape) == 3:
        ax1.imshow(img1)
    else:
        ax1.imshow(img1, cmap='gray')
    ax1.set_title(titles[0], fontsize=ft_size)

    if len(img2.shape) == 3:
        ax2.imshow(img2)
    else:
        ax2.imshow(img2, cmap='gray')
    ax2.set_title(titles[1], fontsize=ft_size)

    plt.suptitle(titles[2], fontsize=ft_size)
    plt.tight_layout()
    if output:
        plt.savefig(output)
        print("Image is saved at {}".format(output))
        plt.close()
    else:
        plt.show()
