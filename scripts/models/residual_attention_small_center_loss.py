import numpy as np

import keras
from keras.models import Model, Sequential
from keras.layers import Input, add, BatchNormalization, Activation, Lambda, GlobalAveragePooling2D, GlobalMaxPooling2D, Embedding
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

import tensorflow as tf
import matplotlib.pyplot as plt


def l2_embedding_loss(y_true, y_pred):
    return y_pred

class ResidualAttentionNetSmallCenterLoss(BaseNetwork):
    def __init__(self, output_directory, checkpoint_directory, config_dict, preprocessor, name = "ResidualAttentionNetSmallCenterLoss", train = True):
        BaseNetwork.__init__(self, output_directory, checkpoint_directory, config_dict, preprocessor, name=name, train = train)
        self.p = int(config_dict['p'])
        self.r = int(config_dict['r'])
        self.t = int(config_dict['t'])
        self.center_loss_strength = float(config_dict['CenterLossStrength']) if 'CenterLossStrength' in config_dict  else 0.5

    def get_network(self):

        inputs = Input(self.preprocessor.get_shape())
        outputs = Lambda(lambda x: (x /255. -0.5) * 2)(inputs)

        outputs = Conv2D(8, (7, 7), strides = [2,2], padding='same', activation = 'relu', name = 'classification_conv_1')(outputs)
        outputs = MaxPooling2D(pool_size=(3,3), strides = [2,2], padding='SAME' , name = 'classification_maxpool_1')(outputs)

        outputs = Residual(8, 16, outputs)
        outputs = ResidualAttention(outputs, p = self.p, t = self.t, r = self.r)
        outputs = MaxPooling2D(pool_size=(3,3), strides = [2,2], padding='SAME' , name = 'classification_maxpool_2')(outputs)
        outputs = Residual(16, 32, outputs)
        outputs = ResidualAttention(outputs, p = self.p, t = self.t, r = self.r)
        outputs = MaxPooling2D(pool_size=(3,3), strides = [2,2], padding='SAME' , name = 'classification_maxpool_3')(outputs) 
        outputs = Residual(32, 64, outputs)
        outputs = ResidualAttention(outputs, p = self.p, t = self.t, r = self.r)
        outputs = MaxPooling2D(pool_size=(3,3), strides = [2,2], padding='SAME' , name = 'classification_maxpool_4')(outputs)
        outputs = Residual(64, 128, outputs)

        outputs = BatchNormalization()(outputs) 
        outputs = GlobalAveragePooling2D()(outputs)
        # outputs = GlobalMaxPooling2D()(outputs)
        outputs = Dense(256, name = 'classification_dense_1', activation='relu')(outputs)
        center_loss_layer = outputs
        outputs = Dense(self.number_of_classes, activation='softmax', name = 'class_prob')(center_loss_layer)

        lambda_c = self.center_loss_strength
        input_target = Input(shape=(1,)) # single value ground truth labels as inputs
        centers = Embedding(self.number_of_classes,256)(input_target)
        l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]),1,keepdims=True),name='l2_loss')([center_loss_layer,centers])
        model = Model(inputs=[inputs,input_target],outputs=[outputs,l2_loss])        
        model.compile(loss=["categorical_crossentropy", l2_embedding_loss],loss_weights=[1,lambda_c], optimizer=Adam(0.0001), metrics=[categorical_accuracy])
        if self.train:
            model.summary()

        return model

    def get_additional_callbacks(self):
        return []#[OutputsCallback(self.transformation_operation, self.preprocessor.X_train, self.preprocessor.IMG_HEIGHT, self.preprocessor.IMG_WIDTH, self.preprocessor.IMG_CHANNELS)] #return array of new callbacks [EarlyStopping(..), ..]
