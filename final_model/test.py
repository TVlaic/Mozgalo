import os

import numpy as np
import cv2
import keras
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input


from ResNetAttentionInceptionBlocksDifferentBnorm import get_network as RNetInceptionBNorm
from ResNetAttentionInceptionBlocks import get_network as RNetInception
from ResNetAttention import get_network as RNet

def rename_layers(model, sufix):
    for i, layer in enumerate(model.layers):
        name = layer.name.split('_')[:-1]
        name = "_".join(name)
        layer.name = name + "_" + sufix + "_" + str(i)
        
def load_img(path):
    x = cv2.imread(path, 0)
    x = cv2.resize(x, (600, 1100))#, mode='constant', preserve_range=True)
    x = x.reshape((1100, 600, 1))
    return x

def get_final_prediction(predictions):
    class_votes = []
    confidence = []
    for result in res:
        class_votes.append(result[0])
        confidence.append(result[1])

    confidence = np.array(confidence)
    class_votes = np.array(class_votes)
    class_name, votes = np.unique(class_votes, return_counts = True)
    max_ind = np.argmax(votes)

    max_vote_indices = np.where(class_votes==class_name[max_ind])
    other_vote_indices = np.where(class_votes!=class_name[max_ind])
    conf_subset = confidence[max_vote_indices]
    conf_oposite_subset = confidence[other_vote_indices]

from sklearn import preprocessing

classes = ['Costco',
 'Meijer',
 'HarrisTeeter',
 'KingSoopers',
 'ShopRite',
 'JewelOsco',
 'SamsClub',
 'HyVee',
 'BJs',
 'Safeway',
 'Target',
 'HEB',
 'Kroger',
 'WholeFoodsMarket',
 'StopShop',
 'FredMeyer',
 'Wegmans',
 'Walmart',
 'Frys',
 'CVSPharmacy',
 'Walgreens',
 'Publix',
 'WinCoFoods',
 'Smiths',
 'Albertsons']

le = preprocessing.LabelEncoder()
le.fit(classes)
# pred = model_ensemble.predict(np.zeros((1,1100,600,1)))

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


from skimage.transform import resize
from skimage.io import imread

import tensorflow as tf
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

class ResidualAttentionNetSmallDifferentInterpolationCenterLoss():
    def __init__(self, ):
#         BaseNetwork.__init__(self, output_directory, checkpoint_directory, config_dict, preprocessor, name=name, train = train)
        self.p = 1
        self.r = 1
        self.t = 2
        self.number_of_classes = 25
        self.center_loss_strength = 0.5
        
    def get_network(self):

        inputs = Input((1100,600,1))
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
        model.compile(loss=["categorical_crossentropy", lambda y_true,y_pred: y_pred],loss_weights=[1,lambda_c], optimizer=Adam(0.0001), metrics=[categorical_accuracy])
        model.summary()

        return model

root = '../inputs/test'
sorting_func = lambda x: int(x.split('/')[-1].split('.')[0])
threshold = 0.95 # 0.95 resnet s center lossom dao 0.901
results = []
confidence = []
original_results = []
cnt_others = 0

# model_ensemble = RNetInceptionBNorm()
model_ensemble = ResidualAttentionNetSmallDifferentInterpolationCenterLoss()
model_ensemble = model_ensemble.get_network()
# rename_layers(model_ensemble, 'seventh')
model_ensemble.load_weights('/home/user/Mozgalo/checkpoints/ResidualAttentionNetSmallDifferentInterpolationCenterLoss/MicroblinkBasePreprocessorImgaugCenterLoss/2018-04-18__18_22_29_0.901/0.0341-0011.hdf5', by_name = True)

for file_name in tqdm(sorted(os.listdir(root), key = sorting_func)):
    full_path = os.path.join(root, file_name)
    image = load_img(full_path)
    plt.imshow(np.squeeze(image))
    plt.show()
    image = np.expand_dims(image, axis=0)
    preds = model_ensemble.predict([image,np.random.rand(1,1)])[0][0]
    print(np.argmax(preds), preds[np.argmax(preds)])
    res = np.array(model_ensemble.predict([image,np.random.rand(1,1)]))#.reshape(7,25)
    print(res)
    max_indices = np.argmax(res, axis=1)
    confidences = res[range(len(max_indices)),max_indices]
    print(max_indices)
    class_names = le.inverse_transform(max_indices)
    print(list(zip(class_names,confidences)))
#     print(res[range(len(max_indices)),max_indices])
#     print(np.argmax(res, axis=1))
#     print(le.inverse_transform([17]))
    break