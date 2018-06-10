import tensorflow as tf
from keras.layers import Input, add, BatchNormalization, Activation, Lambda, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D

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