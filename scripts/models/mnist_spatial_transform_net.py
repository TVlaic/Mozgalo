import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD

from .base_network import BaseNetwork
from .custom_layers.SpatialTransformLayer import SpatialTransformLayer


class MnistSpatialTransformNet(BaseNetwork):
    def __init__(self, output_directory, checkpoint_directory, config_dict, preprocessor, name = "MnistSpatialTransformNet", train = True):
        BaseNetwork.__init__(self, output_directory, checkpoint_directory, config_dict, preprocessor, name=name, train = train)

    def get_network(self):
        # initial weights for localization network
        b = np.zeros((2, 3), dtype='float32')
        b[0, 0] = 1
        b[1, 1] = 1
        W = np.zeros((50, 6), dtype='float32')
        weights = [W, b.flatten()]

        inputs = Input(self.preprocessor.get_shape())
        locnet = MaxPooling2D(pool_size=(2,2))(inputs)
        locnet = Convolution2D(20, (5, 5))(locnet)
        locnet = MaxPooling2D(pool_size=(2,2))(locnet)
        locnet = Convolution2D(20, (5, 5))(locnet)
        locnet = Flatten()(locnet)
        locnet = Dense(50)(locnet)
        locnet = Activation('relu')(locnet)
        locnet = Dense(6, weights=weights)(locnet)
        locnet = Model(inputs = [inputs], outputs = [locnet])


        outputs = SpatialTransformLayer(localization_net=locnet,
                                     output_size=(30,30))(inputs)
        outputs = Convolution2D(32, (3, 3), padding='same')(outputs)
        outputs = Activation('relu')(outputs)
        outputs = MaxPooling2D(pool_size=(2,2))(outputs)
        outputs = Convolution2D(32, (3, 3), padding='same')(outputs)
        outputs = Activation('relu')(outputs)
        outputs = MaxPooling2D(pool_size=(2,2))(outputs)
        outputs = Flatten()(outputs)
        outputs = Dense(256)(outputs)
        outputs = Activation('relu')(outputs)
        outputs = Dense(self.number_of_classes)(outputs)
        outputs = Activation('softmax')(outputs)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.summary()

        return model


    