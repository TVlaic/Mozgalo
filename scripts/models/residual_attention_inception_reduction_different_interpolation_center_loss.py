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
from keras import regularizers
from keras import initializers
from keras.layers.merge import concatenate

from sklearn.metrics import f1_score

from .base_network import BaseNetwork
from .custom_layers.SpatialTransformLayer import SpatialTransformLayer

from skimage.transform import resize
from skimage.io import imread

import tensorflow as tf
import matplotlib.pyplot as plt

def conv2d_bn(x, nb_filter, num_row, num_col,
              padding='same', strides=(1, 1), use_bias=False):
    """
    Utility function to apply conv + BN. 
    (Slightly modified from https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py)
    """
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    x = Conv2D(nb_filter, (num_row, num_col),
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(0.00004),
                      kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(x)
    #x = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(x)
    x = BatchNormalization(axis=channel_axis)(x)#, momentum=0.9997, scale=False)(x)
    x = Activation('relu')(x)
    return x

def block_reduction_a(input, num_output):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    num_filters = num_output//4 #8
    branch_0 = conv2d_bn(input, num_filters, 3, 3, strides=(2,2), padding='valid')

    branch_1 = conv2d_bn(input, num_filters, 1, 1)
    branch_1 = conv2d_bn(branch_1, num_filters, 3, 3)
    branch_1 = conv2d_bn(branch_1, num_filters, 3, 3, strides=(2,2), padding='valid')

    branch_2 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(input)

    x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return x

def block_reduction_b(input, num_output):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    num_filters = num_output//4 #8
    branch_0 = conv2d_bn(input, num_filters, 1, 1)
    branch_0 = conv2d_bn(branch_0, num_filters, 3, 3, strides=(2, 2), padding='valid')

    branch_1 = conv2d_bn(input, num_filters, 1, 1)
    branch_1 = conv2d_bn(branch_1, num_filters, 1, 7)
    branch_1 = conv2d_bn(branch_1, num_filters, 7, 1)
    branch_1 = conv2d_bn(branch_1, num_filters, 3, 3, strides=(2,2), padding='valid')

    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return x


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
        

    size1 = first_residuals._keras_shape[1:3]
    output_soft_mask = MaxPooling2D(pool_size=(2,2), padding='same')(first_residuals) 
    for i in range(r):
        output_soft_mask = Residual(num_channels, num_channels, output_soft_mask)
    
    #skip connection
    output_skip_connection = Residual(num_channels, num_channels, output_soft_mask)
    
    #2r residual blocks and first upsampling 
    size2 = output_soft_mask._keras_shape[1:3]
    output_soft_mask = MaxPooling2D(pool_size=(2,2), padding='same')(output_soft_mask) 
    for i in range(2*r):
        output_soft_mask = Residual(num_channels, num_channels, output_soft_mask)
    # output_soft_mask = UpSampling2D([2, 2])(output_soft_mask)
    output_soft_mask = Lambda(lambda x: tf.image.resize_images(x,
                                                size2,
                                                method=tf.image.ResizeMethod.BILINEAR,
                                                align_corners=False
                                            ))(output_soft_mask)



    #addition of the skip connection
    output_soft_mask = add([output_soft_mask, output_skip_connection])    
    #last r blocks of residuals and upsampling
    for i in range(r):
        output_soft_mask = Residual(num_channels, num_channels, output_soft_mask)
    # output_soft_mask = UpSampling2D([2, 2])(output_soft_mask)
    output_soft_mask = Lambda(lambda x: tf.image.resize_images(x,
                                                size1,
                                                method=tf.image.ResizeMethod.BILINEAR,
                                                align_corners=False
                                            ))(output_soft_mask)
    
    #final attention output
    output_soft_mask = Conv2D(num_channels, (1,1), activation='relu')(output_soft_mask)
    #final attention output
    output_soft_mask = Conv2D(num_channels, (1,1), activation='sigmoid')(output_soft_mask)
    
    output = Lambda(lambda x:(1 + x[0]) * x[1])([output_soft_mask,output_trunk])
    for i in range(p):
        output = Residual(num_channels, num_channels, output)
        
    return output


def l2_embedding_loss(y_true, y_pred):
    return y_pred
    
class ResidualAttentionInceptionReductionNetSmallDifferentInterpolationCenterLoss(BaseNetwork):
    def __init__(self, output_directory, checkpoint_directory, config_dict, preprocessor, name = "ResidualAttentionInceptionReductionNetSmallDifferentInterpolationCenterLoss", train = True):
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
        outputs = BatchNormalization()(outputs)
        # outputs = MaxPooling2D(pool_size=(3,3), strides = [2,2], padding='SAME' , name = 'classification_maxpool_2')(outputs)
        # outputs = Residual(16, 32, outputs)
        outputs = block_reduction_a(outputs, num_output = 32)
        outputs = ResidualAttention(outputs, p = self.p, t = self.t, r = self.r)
        outputs = BatchNormalization()(outputs)
        # outputs = MaxPooling2D(pool_size=(3,3), strides = [2,2], padding='SAME' , name = 'classification_maxpool_3')(outputs) 
        # outputs = Residual(32, 64, outputs)
        outputs = block_reduction_b(outputs, num_output = 64)
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
