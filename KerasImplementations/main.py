"""
compare keras build-in ResNet50 and my ResNet
"""
import keras_resnet
from keras.applications.resnet50 import ResNet50


if __name__ == "__main__":

    num_classes = 1000
    model1 = keras_resnet.build_resnet50_org(num_classes)
    model1.summary()

    model2 = ResNet50(weights=None)
    model2.summary()
