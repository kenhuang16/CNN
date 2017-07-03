import os
import shutil
from skimage.io import imsave
import numpy as np

from data_preparation import load_train_data, load_test_data
from model import UNet


input_shape = (96, 96)


def train_and_predict():
    """"""
    fcn = UNet(input_shape)

    # read and processing training data
    imgs_train, imgs_mask_train = load_train_data(100)
    imgs_train = fcn.preprocessor.fit_transform(imgs_train)

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train = fcn.preprocessor.transform(imgs_mask_train, subtract_mean=False)

    # read and processing testing data
    imgs_test, imgs_id_test = load_test_data(100)
    imgs_test = fcn.preprocessor.transform(imgs_test)

    # train the network
    fcn.model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=10,
                  verbose=1, shuffle=True, validation_split=0.2)

    print('Loading saved weights...')
    # fcn.model.load_weights('weights.h5')

    print('Predicting masks on test data...')
    imgs_mask_test = fcn.model.predict(imgs_test, verbose=1)

    pred_dir = 'predicted'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    else:
        shutil.rmtree(pred_dir)

    print('Saving predicted masks to files...')
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)


if __name__ == '__main__':
    train_and_predict()

