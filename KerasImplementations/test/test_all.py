"""
TODO: test preprocessing_training_data

"""
import os
import unittest

import cv2
import numpy as np

from data_processing import Caltech101, Caltech256
from data_processing import normalize_rgb_images, crop_image, \
    flip_horizontally
from networks import keras_resnet, keras_vgg

ROOT_PATH = os.path.expanduser('~/Projects/datasets')


class TestDataProcessing(unittest.TestCase):
    def test_inplace_normalize_img_batch(self):
        X1 = np.zeros(1020, dtype=float).reshape((51, 2, 2, 5))
        normalize_rgb_images(X1)
        self.assertAlmostEqual(X1.sum(), -1020)

        X2 = (np.ones(1020, dtype=float)*255).reshape((51, 2, 2, 5))
        normalize_rgb_images(X2)
        self.assertAlmostEqual(X2.sum(), 1020)

    def test_inplace_normalize_single_img(self):
        X11 = np.zeros(1020, dtype=float).reshape((51, 4, 5))
        normalize_rgb_images(X11)
        self.assertAlmostEqual(X11.sum(), -1020)

        X22 = (np.ones(1020, dtype=float)*255).reshape((51, 4, 5))
        normalize_rgb_images(X22)
        self.assertAlmostEqual(X22.sum(), 1020)


class TestCaltech101(unittest.TestCase):
    def setUp(self):
        self._data = Caltech101(os.path.join(ROOT_PATH,
                                             'caltech101',
                                             '101_ObjectCategories'))

    def test_class_names(self):
        class_names = self._data.get_class_names()
        self.assertEqual(len(class_names), 101)
        self.assertEqual(class_names[0], 'Faces')

    def test_split_data(self):
        self.assertEqual(len(self._data.images_train), 3030)
        self.assertEqual(len(self._data.labels_train), 3030)
        self.assertEqual(len(self._data.images_test), 5647)
        self.assertEqual(len(self._data.labels_test), 5647)

    def test_crop_image(self):
        img = crop_image(cv2.imread(self._data.images_train[0]), 224)
        self.assertEqual(img.shape, (224, 224, 3))

        img = crop_image(cv2.imread(self._data.images_train[500]), 224)
        self.assertEqual(img.shape, (224, 224, 3))

        img = crop_image(cv2.imread(self._data.images_train[1000]), 224)
        self.assertEqual(img.shape, (224, 224, 3))

        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def test_flip_image_horizontally(self):
        img = flip_horizontally(np.arange(12).reshape(2, 2, 3))
        self.assertEqual(img[1, 0, 0], 0)
        self.assertEqual(img[1, 1, 1], 4)
        self.assertEqual(img[0, 0, 2], 8)


class TestCaltech256(unittest.TestCase):
    def setUp(self):
        self._data = Caltech256(os.path.join(ROOT_PATH,
                                             'caltech256',
                                             '256_ObjectCategories'),
                                n_trains=64)

    def test_class_names(self):
        class_names = self._data.get_class_names()
        self.assertEqual(len(class_names), 256)
        self.assertEqual(class_names[0], '001.ak47')

    def test_split_data(self):
        self.assertEqual(len(self._data.images_train), 16384)
        self.assertEqual(len(self._data.labels_train), 16384)
        self.assertEqual(len(self._data.images_test), 13396)
        self.assertEqual(len(self._data.labels_test), 13396)


class TestBuildResNet(unittest.TestCase):
    def test_build_resnets_not_raise_on_valid_input(self):
        raised = False
        try:
            keras_resnet.build_resnet34_org(1000)
            keras_resnet.build_resnet152_org(1000)
        except:
            raised = True
        self.assertFalse(raised, 'Exception raised')


class TestBuildVgg(unittest.TestCase):
    def test_build_vgg_not_raise_on_valid_input(self):
        raised = False
        try:
            keras_vgg.build_vgg16_org(1000)
            keras_vgg.build_vgg((80, 80, 3), 100, (2, 3, 4), (512, 512),
                                n_filters=64, weight_decay=5e-4,
                                dropout_probas=None,
                                batch_normalization=False)
        except:
            raised = True
        self.assertFalse(raised, 'Exception raised')


if __name__ == "__main__":
    unittest.main()
