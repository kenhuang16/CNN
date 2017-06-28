"""
Functions:
- change_colorspace()
- non_maxima_suppression()
- draw_windows()
- two_plots()

Plot the original and processed image together.

"""
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt


def change_colorspace(img, color_space):
    """Convert the color space of an image

    The read-in image is assumed to have RGB color space.

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
    boxes_array = np.asarray(boxes)
    x1 = np.array(boxes_array[:, 0, 0])
    x2 = np.array(boxes_array[:, 1, 0])
    y1 = np.array(boxes_array[:, 0, 1])
    y2 = np.array(boxes_array[:, 1, 1])

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

        # compute the width and height of the overlap area
        xx1 = np.maximum(x1[i], x1[sorted_index[:-1]])
        yy1 = np.maximum(y1[i], y1[sorted_index[:-1]])
        xx2 = np.minimum(x2[i], x2[sorted_index[:-1]])
        yy2 = np.minimum(y2[i], y2[sorted_index[:-1]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap (the denominator is the smaller area)
        overlap = w*h / np.minimum(area[i], area[sorted_index[:-1]])

        # delete all indices from the index list that have
        sorted_index = np.delete(
            sorted_index, np.concatenate([[len(sorted_index) - 1],
                                          np.where(overlap > threshold)[0]]))

    return pick


def search_cars(img, classifier, scale_ratios=(1.0,), confidence_thresh=0.0,
                overlap_thresh=0.5, step_size=(1.0, 1.0),
                region=((0.0, 0.0), (1.0, 1.0))):
    """Search cars in an image

    :param img: numpy.ndarray
        Original image.
    :param classifier: CarClassifier object
        Classifier.
    :param scale_ratios: float
        Scales of the original image.
    :param confidence_thresh: float
        Threshold of the classification score.
    :param overlap_threshold: float, between 0 and 1
        Overlap threshold when applying non-maxima-suppression.
    :param step_size: 1x2 tuple, float
        Size of the sliding step in the unit of the image shape.
    :param region: tuple of ((x0, y0), (x1, y1))
        Search region of the image in fraction.

    :return car_boxes: list of ((x0, y0), (x1, y1))
        Diagonal coordinates of the predicted boxes.
    """
    x0 = int(region[0][0]*img.shape[1])
    x1 = int(region[1][0]*img.shape[1])
    y0 = int(region[0][1]*img.shape[0])
    y1 = int(region[1][1]*img.shape[0])
    search_region = img[y0:y1, x0:x1, :]

    # Applying sliding window search with different scale ratios
    windows_confidences = []
    windows_coordinates = []
    for ratio in scale_ratios:
        scores, windows = \
            classifier.sliding_window_predict(
                search_region, step_size=step_size, binary=False, scale=ratio)

        windows_confidences.extend(scores)
        for window in windows:
            point1 = (window[0][0] + x0, window[0][1] + y0)
            point2 = (window[1][0] + x0, window[1][1] + y0)
            windows_coordinates.append((point1, point2))

    # pick windows with confidence higher than threshold
    windows_coordinates_threshed = []
    windows_confidences_threshed = []
    for window, confidence in zip(windows_coordinates, windows_confidences):
        if confidence > confidence_thresh:
            windows_coordinates_threshed.append(window)
            windows_confidences_threshed.append(confidence)

    # Apply the 'non maxima suppression' to filter the rest windows
    car_boxes = non_maxima_suppression(windows_coordinates_threshed,
                                       windows_confidences_threshed,
                                       overlap_thresh)

    return car_boxes


def draw_boxes(img, boxes, color=(0, 0, 255), thickness=4):
    """Draw rectangular boxes in an image

    :param img: numpy.ndarray
        Original image.
    :param boxes: list of ((x0, y0), (x1, y1))
        Diagonal coordinates of the boxes.
    :param color: tuple
        RGB color tuple.
    :param thickness: int
        Line thickness.

    :return new_img: numpy.ndarray
        New image with boxes imprinted
    """
    new_img = np.copy(img)
    for box in boxes:
        cv2.rectangle(new_img, box[0], box[1], color, thickness)

    return new_img


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
