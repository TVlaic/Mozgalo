import math

import numpy as np
import cv2

from keras import backend as K
import keras.layers as layers
from keras.layers import activations
from keras.layers import initializers
from keras.layers import regularizers
from keras.layers import constraints
from keras.engine import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.legacy import interfaces
from keras.layers.convolutional import _Conv

def get_gabor_filter_bank(u,v,h,w):
    Kmax = math.pi/2
    f=math.sqrt(2)
    sigma=math.pi
    sqsigma=sigma**2
    postmean = math.exp(-sqsigma/2)
    gfilter_real= np.random.random_sample((u,h,w))
    for i in range(u):
        theta=(i)/u * math.pi
        k= Kmax / (f**(v-1))
        xymax=-1e309
        xymin=1e309
        for y in range(h):
            for x in range(w):
                y1= y-1-((h+1)/2.)
                x1= x-1-((w+1)/2.)
                tmp1=math.exp(-(k*k*(x1*x1+y1*y1)/(2*sqsigma)))
                tmp2=math.cos(k*math.cos(theta)*x1+k*math.sin(theta)*y1)-postmean

                gfilter_real[i][y][x]=k*k*tmp1*tmp2/sqsigma

                if gfilter_real[i][y][x]>xymax:
                       xymax=gfilter_real[i][y][x]
                    
                if gfilter_real[i][y][x]<xymin:
                       xymin=gfilter_real[i][y][x]
                    
        if h != 1:
            for y in range(h):
                for x in range(w):
                    gfilter_real[i][y][x]=(gfilter_real[i][y][x]-xymin)/(xymax-xymin);
         
    return gfilter_real.astype(np.float32)

class GaborConv(_Conv):
    """3D convolution layer (e.g. spatial convolution over volumes).
    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 128, 1)` for 128x128x128 volumes
    with a single channel,
    in `data_format="channels_last"`.
    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 3 integers, specifying the
            depth, height and width of the 3D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 3 integers,
            specifying the strides of the convolution along each spatial dimension.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        gabor_frequency: integer used for calculating gabor wavelets that will
            be used to multiply the convolutional kernels
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `channels_first` corresponds to inputs with shape
            `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 3 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        5D tensor with shape:
        `(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if data_format='channels_first'
        or 5D tensor with shape:
        `(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if data_format='channels_last'.
    # Output shape
        5D tensor with shape:
        `(samples, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if data_format='channels_first'
        or 5D tensor with shape:
        `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)` if data_format='channels_last'.
        `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have changed due to padding.
    """

    @interfaces.legacy_conv3d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 gabor_frequency = 1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GaborConv, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=False, #use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        if self.kernel_size[0] != 1:
            raise Exception("First dimension of the kernel needs to be 1")
        self.input_spec = InputSpec(ndim=5)
        
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        
        self.gabor_kernel = self.create_gabor_kernel(input_shape)
        self.built = True
        
    def create_gabor_kernel(self, input_shape):
        #tensorflow channel ordering
        input_dim = input_shape[-1]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        
        # num_orientations = input_shape[1]
        # gabor_filters = get_gabor_filter_bank(num_orientations, 1, self.kernel_size[1], self.kernel_size[2])
        
        # final_gabor_filter = []
        # for i in range(input_shape[1]):
        #     final_kernel = []
        #     for i in range(self.filters):
        #         output_filter = []
        #         for j in range(input_dim):
        #             channel = []
        #             for orient in range(num_orientations):
        #                 channel.append(K.tf.multiply(gabor_filters[orient], self.kernel[:,:,:,j,i]))
        #             single_channel = K.concatenate(channel,axis=0)
        #             output_filter.append(single_channel)
        #         single_output = K.stack(output_filter,axis=-1)
        #         final_kernel.append(single_output)
        #     gabor_kernel = K.stack(final_kernel,axis=-1)
        #     final_gabor_filter.append(gabor_kernel)
        # final_gabor_filter = K.stack(final_gabor_filter, axis=0)
        
        # return final_gabor_filter # raise Exception("Zivot")


        num_orientations = input_shape[1]
        gabor_filters = get_gabor_filter_bank(num_orientations, 1, self.kernel_size[1], self.kernel_size[2])
        gabor_filters = K.expand_dims(gabor_filters, axis = 3)
        #print(gabor_filters)
        gabor_filters = K.repeat_elements(gabor_filters,input_dim,axis = 3)
        #print(gabor_filters)
        gabor_filters = K.expand_dims(gabor_filters, axis = 4)
        #print(gabor_filters)
        gabor_filters = K.repeat_elements(gabor_filters,self.filters,axis = 4)
        #print(gabor_filters)
        gabor_filters = K.expand_dims(gabor_filters, axis = 0)
        #print(gabor_filters)
        gabor_filters = K.repeat_elements(gabor_filters,num_orientations,axis = 0)
        #print(gabor_filters)
        self.gabor_filters = gabor_filters
        result = K.tf.multiply(self.gabor_filters, self.kernel)
        return result
        
    def call(self, inputs):
        num_convs = self.gabor_kernel.get_shape()[0]
        final_outputs = []
        # print(num_convs)
        # print(self.gabor_kernel.get_shape())
        for i in range(num_convs):
            print("kernel ", self.gabor_kernel[i])
            outputs = K.conv3d(
                    inputs,
                    self.gabor_kernel[i],
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate)
            # print("output ", outputs)
            final_outputs.append(outputs)
            
        outputs = K.concatenate(final_outputs, axis = 1)
        # print(outputs)
        return outputs
            
    def get_config(self):
        config = super(Conv3D, self).get_config()
        config.pop('rank')
        return config