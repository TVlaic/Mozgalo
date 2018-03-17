import os

import numpy as np
from keras.utils import np_utils, generic_utils

from .base_preprocessor import BasePreprocessor

class MultiMnistPreprocessor(BasePreprocessor):
    def __init__(self, input_directory, config_dict, name = "MultiMnistPreprocessor"):
        BasePreprocessor.__init__(self, input_directory, config_dict, name = name)
        self.DATA_PATH = os.path.abspath(os.path.join(self.input_directory, 'mnist_cluttered.npz'))

        #np.random.seed(10)

    def load_train(self):
        data = np.load(self.DATA_PATH)
        nb_classes = data['y_train'].shape[1]
        X_train, y_train = data['x_train'], np.argmax(data['y_train'], axis=-1)
        if self.IMG_CHANNELS == 1:
            X_train = X_train.reshape((X_train.shape[0], self.IMG_HEIGHT, self.IMG_WIDTH, 1))
        else:
            X_train = np.dstack([X_train] * self.IMG_CHANNELS)

        y_train = np_utils.to_categorical(y_train, nb_classes)
        print("Train shape %s, %s" % (X_train.shape, y_train.shape))

        self.X_train = X_train
        self.y_train = y_train



    def load_validation(self):
        data = np.load(self.DATA_PATH)
        nb_classes = data['y_valid'].shape[1]
        X_valid, y_valid = data['x_valid'], np.argmax(data['y_valid'], axis=-1)
        if self.IMG_CHANNELS == 1:
            X_valid = X_valid.reshape((X_valid.shape[0], self.IMG_HEIGHT, self.IMG_WIDTH, 1))
        else:
            X_valid = np.dstack([X_valid] * self.IMG_CHANNELS)

        y_valid = np_utils.to_categorical(y_valid, nb_classes)
        print("Valid shape %s, %s" % (X_valid.shape, y_valid.shape))

        self.X_valid = X_valid
        self.y_valid = y_valid



    def load_test(self):
        data = np.load(self.DATA_PATH)
        nb_classes = data['y_test'].shape[1]
        X_test, y_test = data['x_test'], np.argmax(data['y_test'], axis=-1)
        # reshape for convolutions
        if self.IMG_CHANNELS == 1:
            X_test = X_test.reshape((X_test.shape[0], self.IMG_HEIGHT, self.IMG_WIDTH, 1))
        else:
            X_test = np.dstack([X_test] * self.IMG_CHANNELS)

        y_test = np_utils.to_categorical(y_test, nb_classes)
        print("Test shape %s, %s" % (X_test.shape, y_test.shape))

        self.X_test = X_test
        self.y_test = y_test


    def get_train(self):
        return self.X_train, self.y_train

    def get_validation(self):
        return self.X_valid, self.y_valid

    def get_test(self):
        return self.X_test, self.y_test

