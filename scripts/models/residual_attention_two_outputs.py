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
from keras.losses import binary_crossentropy, categorical_crossentropy
import keras.backend as K
import keras.layers as KL

from sklearn.metrics import f1_score

from .base_network import BaseNetwork
from .custom_layers.SpatialTransformLayer import SpatialTransformLayer

from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt

def conv_block(feat_maps_out, prev):
    prev = BatchNormalization()(prev) # Specifying the axis and mode allows for later merging
    prev = Activation('relu')(prev)
    prev = Conv2D(feat_maps_out, (3, 3), padding='same')(prev) 
    prev = BatchNormalization()(prev) # Specifying the axis and mode allows for later merging
    prev = Activation('relu')(prev)
    prev = Conv2D(feat_maps_out, (3, 3), padding='same')(prev) 
    return prev


def skip_block(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = Conv2D(feat_maps_out, (1, 1), padding='same')(prev)
    return prev 


def Residual(feat_maps_in, feat_maps_out, prev_layer):
    '''
    A customizable residual unit with convolutional and shortcut blocks
    Args:
      feat_maps_in: number of channels/filters coming in, from input or previous layer
      feat_maps_out: how many output channels/filters this block will produce
      prev_layer: the previous layer
    '''

    skip = skip_block(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block(feat_maps_out, prev_layer)
    return add([skip, conv]) # the residual connection

def ResidualAttention(inputs, p = 1, t = 2, r = 1):
    channel_axis = -1
    num_channels = inputs._keras_shape[channel_axis]
    
    first_residuals = inputs
    for i in range(p):
        first_residuals = Residual(num_channels, num_channels, first_residuals)
        
    output_trunk = first_residuals    
    for i in range(t):
        output_trunk = Residual(num_channels, num_channels, output_trunk)
        

    output_soft_mask = MaxPooling2D(pool_size=(2,2), padding='same')(first_residuals) 
    for i in range(r):
        output_soft_mask = Residual(num_channels, num_channels, output_soft_mask)
    
    #skip connection
    output_skip_connection = Residual(num_channels, num_channels, output_soft_mask)
    
    #2r residual blocks and first upsampling
    output_soft_mask = MaxPooling2D(pool_size=(2,2), padding='same')(output_soft_mask)  
    for i in range(2*r):
        output_soft_mask = Residual(num_channels, num_channels, output_soft_mask)
    output_soft_mask = UpSampling2D([2, 2])(output_soft_mask)


    #addition of the skip connection
    output_soft_mask = add([output_soft_mask, output_skip_connection])         
    
    #last r blocks of residuals and upsampling
    for i in range(r):
        output_soft_mask = Residual(num_channels, num_channels, output_soft_mask)
    output_soft_mask = UpSampling2D([2, 2])(output_soft_mask)
    
    #final attention output
    output_soft_mask = Conv2D(num_channels, (1,1), activation='relu')(output_soft_mask)
    #final attention output
    output_soft_mask = Conv2D(num_channels, (1,1), activation='sigmoid')(output_soft_mask)
    
    output = Lambda(lambda x:(1 + x[0]) * x[1])([output_soft_mask,output_trunk])
    for i in range(p):
        output = Residual(num_channels, num_channels, output)
        
    return output


class ResidualAttentionTwoOutputsNet(BaseNetwork):
    def __init__(self, output_directory, checkpoint_directory, config_dict, preprocessor, name = "ResidualAttentionTwoOutputsNet", train = True):
        BaseNetwork.__init__(self, output_directory, checkpoint_directory, config_dict, preprocessor, name=name, train = train)
        self.p = int(config_dict['p'])
        self.r = int(config_dict['r'])
        self.t = int(config_dict['t'])

    def get_network(self):

        inputs = Input(self.preprocessor.get_shape())

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
        outputs1 = Dense(self.number_of_classes, activation='softmax', name = 'class')(outputs)

        outputs2 = Dense(1, activation='sigmoid', name = 'real_image')(outputs)

        model = Model(inputs=[inputs], outputs=[outputs1, outputs2])

        loss = {
              'class': categorical_crossentropy,
              'real_image': binary_crossentropy
            }
        loss_weights = {
              'class': 1,
              'real_image': 1,
            }
        model.compile(loss=loss, loss_weights=loss_weights, optimizer=Adam(0.0001))#, metrics=[categorical_accuracy])
        model.summary()

        # X_in = model.input
        # X_transformed = model.layers[1].output
        # print(model.layers[1].name)
        # self.transformation_operation = K.function([X_in], [X_transformed])

        return model

    def get_additional_callbacks(self):
        return []#[OutputsCallback(self.transformation_operation, self.preprocessor.X_train, self.preprocessor.IMG_HEIGHT, self.preprocessor.IMG_WIDTH, self.preprocessor.IMG_CHANNELS)] #return array of new callbacks [EarlyStopping(..), ..]
