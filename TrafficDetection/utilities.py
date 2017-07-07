"""
Various help functions:
- change_colorspace()
- read_image_data()
- augment_image_data()
- get_perspective_trans_matrix()
- non_maxima_suppression()
- merge_box()
- box_by_heat()
- draw_box()
- two_plots()

Plot the original and processed image together.

"""
import re
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label

from parameters import car_files, noncar_files


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


def read_image_data(max_count=100000):
    """Read image data from files.

    :param max_count: int
        Maximum number of data to read.

    :returns: numpy.ndarray
        Image data.
    :return labels: 1D numpy.ndarray
        Image labels.
    """
    imgs = []
    labels = np.hstack((np.ones(len(car_files)),
                        np.zeros(len(noncar_files)))).astype(np.int8)

    for file in car_files + noncar_files:
        img = cv2.imread(file)
        imgs.append(img)

        if len(imgs) > max_count:
            break

    return np.array(imgs, dtype=np.uint8), labels


def augment_image_data(imgs, labels, number=5000, gain=0.3, bias=30):
    """Augment image data

    @param imgs: numpy.ndarray
        Features.
    @param labels: 1D numpy.ndarray
        Labels.
    @param number: int
        Number of augmented data.
    @param gain: float
        Maximum gain jitter.
    @param bias: int
        Maximum absolute bias jitter.

    @return new_features: numpy.ndarray
        Augmented features.
    @return new_labels: 1D numpy.ndarray
        Augmented labels.
    """
    new_imgs = []
    new_labels = []

    size = len(imgs)
    for i in range(number):
        choice = random.randint(0, size - 1)
        img = imgs[choice]

        # Randomly flip the image horizontally
        if random.random() > 0.5:
            img = cv2.flip(img, 1)

        # Randomly remove the color information
        if random.random() > 0.5:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.stack([img_gray, img_gray, img_gray], axis=-1)

        # Random brightness and contrast jitter
        if img.mean() > 155:
            alpha = 1 + gain*(random.random() - 1.0)
            beta = bias*(random.random() - 1.0)
        elif img.mean() < 100:
            alpha = 1 + gain*(1.0 - random.random())
            beta = bias*(1.0 - random.random())
        else:
            alpha = 1 + gain * (2 * random.random() - 1.0)
            beta = bias * (2 * random.random() - 1.0)
        img = img*alpha + beta
        img[img > 255] = 255
        img[img < 0] = 0

        new_imgs.append(img)
        new_labels.append(labels[choice])

    return np.asarray(new_imgs, dtype=np.uint8), np.asarray(new_labels)


def get_perspective_trans_matrix(src, dst):
    """Get the perspective and inverse perspective transfer matrices

    :param src:

    :param dst:

    :return perspective transform matrix and inverse perspective
            transform matrix
    """
    ppt_trans_matrix = cv2.getPerspectiveTransform(src, dst)
    inv_ppt_trans_matrix = cv2.getPerspectiveTransform(dst, src)

    return ppt_trans_matrix, inv_ppt_trans_matrix


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

    :return a list of diagonal coordinates ((x0, y0), (x1, y1))
            of the selected boxes.
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

        # compute the ratio of overlap
        overlap = w*h / area[sorted_index[:-1]]

        # delete all indices from the index list that have
        sorted_index = np.delete(
            sorted_index, np.concatenate([[len(sorted_index) - 1],
                                          np.where(overlap > threshold)[0]]))

    return pick


def merge_box(boxes, shape):
    """Merge connected boxes

    :param boxes: list of tuples
        A list of diagonal coordinates ((x0, y0), (x1, y1)) of boxes.
    :param shape: tuple, (x, y)
        Shape of the background image.

    :return new_boxes: list of tuples
        A list of diagonal coordinates ((x0, y0), (x1, y1)) of boxes.
    """
    if len(boxes) == 0:
        return []

    # construct the background
    bkg = np.zeros(shape)
    # Assigned 1 to the areas enclosed by the boxes (including border)
    for box in boxes:
        for x in range(box[0][0], box[1][0] + 1):
            for y in range(box[0][1], box[1][1] + 1):
                bkg[y, x] = 1

    # Label the connected area
    labeled, count = label(bkg, connectivity=1, return_num=True)

    # Make new boxes according to the labels
    new_boxes = []
    for i in range(count):
        indices = np.where(labeled == i + 1)
        new_boxes.append(((min(indices[1]), min(indices[0])),
                          (max(indices[1]), max(indices[0]))))

    return new_boxes


def box_by_heat(boxes, confidences, shape, thresh=1.0):
    """Select the boxes by 'heat' (sum of confidence)

    :param boxes: list of tuples
        A list of diagonal coordinates ((x0, y0), (x1, y1)) of boxes.
    :param confidences: list
        Confidences of the boxes.
    :param shape: tuple, (x, y)
        Shape of the background image.
    :param thresh: float
        Threshold of the heat map.

    :return new_boxes: list of tuples
        A list of diagonal coordinates ((x0, y0), (x1, y1)) of boxes.
    """
    if len(boxes) == 0:
        return []

    # construct the background
    heat_map = np.zeros(shape)
    # calculate the heat of each pixel by adding up the confidences
    for i in range(len(boxes)):
        for x in range(boxes[i][0][0], boxes[i][1][0] + 1):
            for y in range(boxes[i][0][1], boxes[i][1][1] + 1):
                heat_map[y, x] += confidences[i]

    heat_map[heat_map < thresh] = 0

    # Label the connected area
    heat_map[heat_map > 0] = 1
    labeled, count = label(heat_map, connectivity=1, return_num=True)

    # Make new boxes according to the labels
    new_boxes = []
    for i in range(count):
        indices = np.where(labeled == i + 1)
        new_boxes.append(((min(indices[1]), min(indices[0])),
                          (max(indices[1]), max(indices[0]))))

    return new_boxes


def draw_box(img, boxes, color=(0, 0, 255), thickness=4):
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
