import numpy as np

import keras
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, MaxPooling3D
from keras.layers.core import Dropout, Lambda
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.metrics import categorical_accuracy
import keras.backend as K
import keras.layers as KL

from sklearn.metrics import f1_score

from .base_network import BaseNetwork
from .custom_layers.GaborConv import GaborConv
from .custom_layers.SpatialTransformLayer import SpatialTransformLayer

from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt


class GaborConvolutionNet(BaseNetwork):
    def __init__(self, output_directory, checkpoint_directory, config_dict, preprocessor, name = "GaborConvolutionNet", train = True):
        BaseNetwork.__init__(self, output_directory, checkpoint_directory, config_dict, preprocessor, name=name, train = train)
        y, x = config_dict['STNOutputSize'].split(',')
        x = int(x)
        y = int(y)
        self.STN_output_size = (y, x)
        self.train_localization = bool(config_dict['TrainLocalization'])

    def get_localization_network(self, inputs):
        # initial weights for localization network for identity transform
        b = np.zeros((2, 3), dtype='float32')
        b[0, 0] = 1
        b[1, 1] = 1 
        W = np.zeros((50, 6), dtype='float32')
        weights = [W, b.flatten()]

        locnet = GaborConv(16, (1,11, 11), gabor_frequency = 1, activation='relu', name = 'localization_conv_1')(inputs)
        locnet = MaxPooling3D(pool_size=(1,2,2), name = 'localization_maxpool_1')(locnet)

        locnet = GaborConv(32, (1,7, 7), gabor_frequency = 2, activation='relu', name = 'localization_conv_3')(locnet)
        locnet = MaxPooling3D(pool_size=(1,2,2), name = 'localization_maxpool_2')(locnet)

        locnet = GaborConv(64, (1,5, 5), gabor_frequency = 3, activation='relu', name = 'localization__conv_5')(locnet)
        locnet = MaxPooling3D(pool_size=(1,2,2), name = 'localization_maxpool_3')(locnet)

        locnet = GaborConv(32, (1,3, 3), gabor_frequency = 4, activation='relu', name = 'localization_conv_7')(locnet)
        locnet = MaxPooling3D(pool_size=(1,2,2), name = 'localization_maxpool_4')(locnet)

        locnet = GaborConv(16, (1,3, 3), gabor_frequency = 5, activation='relu', name = 'localization_conv_9')(locnet)
        locnet = MaxPooling3D(pool_size=(1,2,2), name = 'localization_maxpool_5')(locnet)

        locnet = Flatten()(locnet)
        locnet = Dense(50, activation = 'relu', name = 'localization_dense_1')(locnet)
        locnet = Dense(6, weights=weights, name = 'localization_dense_affine')(locnet)
        locnet = Model(inputs = [inputs], outputs = [locnet])

        return locnet

    def get_network(self):

        inputs = Input(self.preprocessor.get_shape())

        # self.locnet = self.get_localization_network(inputs)
        
        # for layer in self.locnet.layers:
        #     layer.trainable = self.train_localization

        # outputs = SpatialTransformLayer(localization_net=self.locnet,
        #                              output_size=self.STN_output_size, name='spatial_layer_1')(inputs)


        # s = Lambda(lambda x: x / 255.) (outputs)

        outputs = GaborConv(32, (1,3, 3), gabor_frequency = 1, padding='valid', activation = 'relu', name = 'classification_conv_1')(inputs)
        outputs = MaxPooling3D(pool_size=(1,2,2), name = 'classification_maxpool_1')(outputs)

        outputs = GaborConv(32, (1,3, 3), gabor_frequency = 2, padding='valid', activation = 'relu', name = 'classification_conv_2')(outputs)
        outputs = MaxPooling3D(pool_size=(1,2,2), name = 'classification_maxpool_2')(outputs)

        outputs = GaborConv(32, (1,3, 3), gabor_frequency = 3, padding='valid', activation = 'relu', name = 'classification_conv_3')(outputs)
        outputs = MaxPooling3D(pool_size=(1,2,2), name = 'classification_maxpool_3')(outputs)

        outputs = GaborConv(32, (1,3, 3), gabor_frequency = 4, padding='valid', activation = 'relu', name = 'classification_conv_4')(outputs)
        outputs = MaxPooling3D(pool_size=(1,2,2), name = 'classification_maxpool_4')(outputs)

        outputs = GaborConv(32, (1,3, 3), gabor_frequency = 5, padding='valid', activation = 'relu', name = 'classification_conv_5')(outputs)
        outputs = MaxPooling3D(pool_size=(1,2,2), name = 'classification_maxpool_5')(outputs)

        outputs = GaborConv(32, (1,3, 3), gabor_frequency = 6, padding='valid', activation = 'relu', name = 'classification_conv_6')(outputs)
        outputs = MaxPooling3D(pool_size=(1,2,2), name = 'classification_maxpool_6')(outputs)

        outputs = Flatten()(outputs)
        outputs = Dense(256, activation = 'relu', name = 'classification_dense_1')(outputs)
        outputs = Dense(self.number_of_classes, activation='softmax', name = 'classification_dense_probs')(outputs)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])
        model.summary()

        return model

    def get_additional_callbacks(self):
        return []#[OutputsCallback(self.transformation_operation, self.preprocessor.X_train, self.preprocessor.IMG_HEIGHT, self.preprocessor.IMG_WIDTH, self.preprocessor.IMG_CHANNELS)] #return array of new callbacks [EarlyStopping(..), ..]
