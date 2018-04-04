import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, concatenate
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD
from keras.losses import binary_crossentropy
import keras.backend as K

from .base_network import BaseNetwork
from .custom_layers.SpatialTransformLayer import SpatialTransformLayer

import tensorflow as tf


def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def class_wise_binary_entropy(y_true, y_pred):
    number_of_classes = K.int_shape(y_pred)[1]
    loss = 0
    weighted_binary_crossentropy = create_weighted_binary_crossentropy(1./number_of_classes, (number_of_classes-1.)/number_of_classes)
    for i in range(number_of_classes):
        #loss += -tf.reduce_sum(y_true[:,i] * tf.log(y_pred[:,i]), axis=1)
        #loss += binary_crossentropy(y_true[:,i], y_pred[:,i])
        loss += weighted_binary_crossentropy(y_true[:,i], y_pred[:,i])
    return loss

class MultiBranchMnist(BaseNetwork):
    def __init__(self, output_directory, checkpoint_directory, config_dict, preprocessor, name = "MultiBranchMnist", train = True):
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


        spatial = SpatialTransformLayer(localization_net=locnet,
                                     output_size=(30,30))(inputs)
        final_output = None
        array = []
        for i in range(self.number_of_classes):
            # locnet = MaxPooling2D(pool_size=(2,2))(inputs)
            # locnet = Convolution2D(20, (5, 5))(locnet)
            # locnet = MaxPooling2D(pool_size=(2,2))(locnet)
            # locnet = Convolution2D(20, (5, 5))(locnet)
            # locnet = Flatten()(locnet)
            # locnet = Dense(50)(locnet)
            # locnet = Activation('relu')(locnet)
            # locnet = Dense(6, weights=weights)(locnet)
            # locnet = Model(inputs = [inputs], outputs = [locnet])


            # outputs = SpatialTransformLayer(localization_net=locnet,
            #                              output_size=(30,30))(inputs)

            # outputs = Convolution2D(32, (3, 3), padding='same')(outputs)
            outputs = Convolution2D(32, (3, 3), padding='same')(spatial)
            outputs = Activation('relu')(outputs)
            outputs = MaxPooling2D(pool_size=(2,2))(outputs)
            outputs = Convolution2D(32, (3, 3), padding='same')(outputs)
            outputs = Activation('relu')(outputs)
            outputs = MaxPooling2D(pool_size=(2,2))(outputs)
            outputs = Flatten()(outputs)
            outputs = Dense(256)(outputs)
            outputs = Activation('relu')(outputs)
            outputs = Dense(1)(outputs)
            outputs = Activation('sigmoid')(outputs)
            array.append(outputs)
            #if final_output == None:
            #    final_output = outputs
            #else:
            #    final_output = concatenate([final_output, outputs])
        final_output = concatenate(array)      
        model = Model(inputs=[inputs], outputs=[final_output])
        model.compile(loss=class_wise_binary_entropy, optimizer='adam')
        model.summary()

        return model


    