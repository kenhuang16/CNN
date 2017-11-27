"""

"""
from keras_resnet import build_resnet
from keras_resnet import build_resnet18_org
from keras_resnet import build_resnet34_org
from keras_resnet import build_resnet50_org
from keras_resnet import build_resnet101_org
from keras_resnet import build_resnet152_org


if __name__ == "__main__":

    # input_shape = (224, 224, 3)
    num_classes = 10

    model1 = build_resnet18_org(num_classes)
    model1.summary()

    model2 = build_resnet50_org(num_classes)
    model2.summary()

    input_shape = (32, 32, 3)
    model3 = build_resnet(input_shape, num_classes, stage_blocks=(5, 5, 5),
                          filters=16, bottleneck=False)
    model3.summary()
