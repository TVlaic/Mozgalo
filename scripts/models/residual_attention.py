import numpy as np

import keras
from keras.models import Model, Sequential
from keras.layers import Input, add, BatchNormalization, Activation, Lambda, GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Dropout, Lambda
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.metrics import categorical_accuracy
import keras.backend as K
import keras.layers as KL

from sklearn.metrics import f1_score

from .base_network import BaseNetwork
from .custom_layers.residual_layers import *

from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt

class ResidualAttentionNet(BaseNetwork):
    def __init__(self, output_directory, checkpoint_directory, config_dict, preprocessor, name = "ResidualAttentionNet", train = True):
        BaseNetwork.__init__(self, output_directory, checkpoint_directory, config_dict, preprocessor, name=name, train = train)
        self.p = int(config_dict['p'])
        self.r = int(config_dict['r'])
        self.t = int(config_dict['t'])

    def get_network(self):

        inputs = Input(self.preprocessor.get_shape())
        outputs = Lambda(lambda x: (x /255. -0.5) * 2)(inputs)

        outputs = Conv2D(32, (7, 7), strides = [1,1], padding='same', activation = 'relu', name = 'classification_conv_1')(inputs)
        outputs = MaxPooling2D(pool_size=(3,3), strides = [2,2], padding='SAME' , name = 'classification_maxpool_1')(outputs)

        outputs = Residual(32, 64, outputs)
        outputs = ResidualAttention(outputs, p = self.p, t = self.t, r = self.r)
        outputs = MaxPooling2D(pool_size=(3,3), strides = [2,2], padding='SAME' , name = 'classification_maxpool_2')(outputs)
        outputs = Residual(64, 128, outputs)
        outputs = ResidualAttention(outputs, p = self.p, t = self.t, r = self.r)
        outputs = MaxPooling2D(pool_size=(3,3), strides = [2,2], padding='SAME' , name = 'classification_maxpool_3')(outputs) 
        outputs = Residual(128, 256, outputs)
        outputs = ResidualAttention(outputs, p = self.p, t = self.t, r = self.r)
        outputs = MaxPooling2D(pool_size=(3,3), strides = [2,2], padding='SAME' , name = 'classification_maxpool_4')(outputs)
        outputs = Residual(256, 512, outputs)

        outputs = GlobalAveragePooling2D()(outputs)
        outputs = Dense(256, activation = 'relu', name = 'classification_dense_1')(outputs)
        outputs = Dense(self.preprocessor.get_number_of_classes(), activation='softmax', name = 'classification_dense_probs')(outputs)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=[categorical_accuracy])
        model.summary()

        # X_in = model.input
        # X_transformed = model.layers[1].output
        # print(model.layers[1].name)
        # self.transformation_operation = K.function([X_in], [X_transformed])

        return model

    def get_additional_callbacks(self):
        return []#[OutputsCallback(self.transformation_operation, self.preprocessor.X_train, self.preprocessor.IMG_HEIGHT, self.preprocessor.IMG_WIDTH, self.preprocessor.IMG_CHANNELS)] #return array of new callbacks [EarlyStopping(..), ..]
