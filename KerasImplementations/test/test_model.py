import os
import unittest

from networks import keras_resnet, keras_vgg

ROOT_PATH = os.path.expanduser('~/Projects/datasets')


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
