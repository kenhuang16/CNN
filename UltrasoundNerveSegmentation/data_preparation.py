"""
Created on 18/06/2017
Prepare training and testing data
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread


data_path = 'data/'
train_data_file = os.path.join(data_path, 'imgs_train.npy')
train_mask_data_file = os.path.join(data_path, 'imgs_mask_train.npy')
test_data_file = os.path.join(data_path, 'imgs_test.npy')
test_id_data_file = os.path.join(data_path, 'imgs_id_test.npy')

image_height = 420
image_width = 580


def create_train_data():
    """Create training data

    The image data and the image mask data.
    """
    train_data_path = os.path.join(data_path, 'train')
    images = glob.glob(os.path.join(train_data_path, "*.tif"))

    image_count = int(len(images) / 2)  # image and its mask
    assert(image_count == 5635)

    imgs = np.ndarray((image_count, image_height, image_width, 1), dtype=np.uint8)
    imgs_mask = np.ndarray((image_count, image_height, image_width, 1), dtype=np.uint8)

    i = 0
    print('Reading training images...')
    for image_file in images:
        if 'mask' in image_file:
            continue
        image_mask_name = image_file.split('.')[0] + '_mask.tif'
        img = imread(image_file)
        img_mask = imread(image_mask_name)

        imgs[i, :, :, 0] = img
        imgs_mask[i, :, :, 0] = img_mask

        i += 1

    np.save(train_data_file, imgs)
    np.save(train_mask_data_file, imgs_mask)
    print('Training image data were saved in {} and {}.'.
          format(train_data_file, train_mask_data_file))


def create_test_data():
    """Create testing data

    The image data and the image id (number in the file name)
    """
    test_data_path = os.path.join(data_path, 'test')
    images = glob.glob(os.path.join(test_data_path, "*.tif"))

    image_count = len(images)
    assert(image_count == 5508)

    imgs = np.ndarray((image_count, image_height, image_width, 1), dtype=np.uint8)
    imgs_id = np.ndarray((image_count, ), dtype=np.int32)

    i = 0
    print('Reading testing images...')
    for image_file in images:
        img_id = int(image_file.split('/')[-1].split('.')[0])
        img = imread(image_file)

        imgs[i, :, :, 0] = img
        imgs_id[i] = img_id

        i += 1

    np.save(test_data_file, imgs)
    np.save(test_id_data_file, imgs_id)
    print('Testing image data were saved in {} and {}.'.
          format(test_data_file, test_id_data_file))


def load_train_data(count=None):
    """Load train image data

    :param count: None/int
        Number of images to load.

    :return imgs_train: numpy.ndarray
        Training Images
    :return imgs_mask_train: numpy.ndarray
        Images with only mask.
    """
    if not os.path.isfile(train_data_file) or not os.path.isfile(train_mask_data_file):
        create_train_data()

    if count is None:
        imgs_train = np.load(train_data_file)
        imgs_mask_train = np.load(train_mask_data_file)
    else:
        imgs_train = np.load(train_data_file)[:count]
        imgs_mask_train = np.load(train_mask_data_file)[:count]

    print("Loaded {} training image data from {}".
          format(imgs_train.shape[0], train_data_file))
    print("Loaded {} training image mask data from {}".
          format(imgs_mask_train.shape[0], train_mask_data_file))

    return imgs_train, imgs_mask_train


def load_test_data(count=None):
    """Load test image data

    :param count: None/int
        Number of images to load.

    :return imgs_test: numpy.ndarray
        Test Images.
    """
    if not os.path.isfile(test_data_file) or not os.path.isfile(test_id_data_file):
        create_test_data()

    if count is None:
        imgs_test = np.load(test_data_file)
        imgs_id = np.load(test_id_data_file)
    else:
        imgs_test = np.load(test_data_file)[:count]
        imgs_id = np.load(test_id_data_file)[:count]

    print("Loaded {} testing image data from {}".
          format(imgs_test.shape[0], test_data_file))

    return imgs_test, imgs_id


if __name__ == "__main__":

    img_train = imread(os.path.join(data_path, 'train/1_1.tif'))
    img_mask_train = imread(os.path.join(data_path, 'train/1_1_mask.tif'))
    img_test = imread(os.path.join(data_path, 'test/1.tif'))

    print("Shape of image: {}".format(img_train.shape))
    assert(img_train.shape == img_mask_train.shape == img_test.shape)

    img_train_with_mask = img_train - 0.2*img_mask_train
    img_train_with_mask[img_train_with_mask < 0] = 0

    plt.imshow(img_train_with_mask.astype(np.uint8), cmap='gray')
    plt.title("Training image with mask")
    plt.show()
    plt.imshow(img_test, cmap='gray')
    plt.title("Testing image")
    plt.show()