import numpy as np

import keras
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dropout, Lambda
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.metrics import categorical_accuracy
import keras.backend as K
import keras.layers as KL

from sklearn.metrics import f1_score

from .base_network import BaseNetwork
from .custom_layers.SpatialTransformLayer import SpatialTransformLayer

from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt

class OutputsCallback(keras.callbacks.Callback):
    def __init__(self, transform_function, img_paths, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        self.F = transform_function
        self.paths = img_paths
        self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS = IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS


    def on_batch_begin(self, batch, logs={}):
        # import matplotlib.pyplot as plt
        if batch%50 != 0:
            return


        X1 = np.zeros((6,self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        indices = np.random.randint(0, self.paths.shape[0], 3)
        for i, ind in enumerate(indices):
            # image = imread(self.paths[ind])
            # image = resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH))
            # if len(image.shape) != 3:
            #     image = np.dstack([image,image,image])


            image = imread(self.paths[ind], as_grey = self.IMG_CHANNELS==1)
            image = resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH))
            if len(image.shape) != 3 and self.IMG_CHANNELS == 3:
                image = np.dstack([image,image,image])
                image = image/255.
            elif self.IMG_CHANNELS == 1:
                image = image.reshape((self.IMG_HEIGHT, self.IMG_WIDTH, 1))    

            X1[i] = image

        Xresult = self.F([X1])
        # plt.clf()
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i+1)
            if i>=3:
                image = X1[i-3]
            else:
                image = (Xresult[0][i]*255).astype(np.uint8)

            if image.shape[2]==1:
                plt.imshow(np.squeeze(image))
            else:
                plt.imshow(image)
            plt.axis('off')
        plt.savefig('../outputs/MicroblinkBaseNet/%s.png' % batch)
        plt.close()


class MicroblinkBaseNet(BaseNetwork):
    def __init__(self, output_directory, checkpoint_directory, config_dict, preprocessor, name = "MicroblinkBaseNet", train = True):
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


        # initial weights for localization network zoom on top of image
        # b = np.zeros((2, 3), dtype='float32')
        # b[0, 0] = 1
        # b[1, 1] = 0.3
        # b[1, 2] = -0.7
        # W = np.zeros((50, 6), dtype='float32')
        # weights = [W, b.flatten()]

        # s = Lambda(lambda x: x / 255.) (inputs)

        locnet = Convolution2D(16, (11, 11), activation='relu', name = 'localization_conv_1')(inputs)
        locnet = Convolution2D(16, (1, 11), activation='relu', name = 'localization_conv_2')(inputs)
        locnet = MaxPooling2D(pool_size=(2,2), name = 'localization_maxpool_1')(locnet)

        locnet = Convolution2D(32, (7, 7), activation='relu', name = 'localization_conv_3')(locnet)
        locnet = Convolution2D(32, (7, 7), activation='relu', name = 'localization_conv_4')(locnet)
        locnet = MaxPooling2D(pool_size=(2,2), name = 'localization_maxpool_2')(locnet)

        locnet = Convolution2D(64, (5, 5), activation='relu', name = 'localization__conv_5')(locnet)
        locnet = Convolution2D(64, (5, 5), activation='relu', name = 'localization_conv_6')(locnet)
        locnet = MaxPooling2D(pool_size=(2,2), name = 'localization_maxpool_3')(locnet)

        locnet = Convolution2D(32, (3, 3), activation='relu', name = 'localization_conv_7')(locnet)
        locnet = Convolution2D(32, (3, 3), activation='relu', name = 'localization_conv_8')(locnet)
        locnet = MaxPooling2D(pool_size=(2,2), name = 'localization_maxpool_4')(locnet)

        locnet = Convolution2D(16, (3, 3), activation='relu', name = 'localization_conv_9')(locnet)
        locnet = Convolution2D(16, (3, 3), activation='relu', name = 'localization_conv_10')(locnet)
        locnet = MaxPooling2D(pool_size=(2,2), name = 'localization_maxpool_5')(locnet)

        locnet = Flatten()(locnet)
        locnet = Dense(50, activation = 'relu', name = 'localization_dense_1')(locnet)
        locnet = Dense(6, weights=weights, name = 'localization_dense_affine')(locnet)
        locnet = Model(inputs = [inputs], outputs = [locnet])

        return locnet


    def get_network(self):

        inputs = Input(self.preprocessor.get_shape())

        self.locnet = self.get_localization_network(inputs)
        
        for layer in self.locnet.layers:
            layer.trainable = self.train_localization

        outputs = SpatialTransformLayer(localization_net=self.locnet,
                                     output_size=self.STN_output_size, name='spatial_layer_1')(inputs)


        # s = Lambda(lambda x: x / 255.) (outputs)

        outputs = Convolution2D(32, (3, 3), padding='same', activation = 'relu', name = 'classification_conv_1')(outputs)
        outputs = MaxPooling2D(pool_size=(2,2), name = 'classification_maxpool_1')(outputs)

        outputs = Convolution2D(32, (3, 3), padding='same', activation = 'relu', name = 'classification_conv_2')(outputs)
        outputs = MaxPooling2D(pool_size=(2,2), name = 'classification_maxpool_2')(outputs)

        outputs = Convolution2D(32, (3, 3), padding='same', activation = 'relu', name = 'classification_conv_3')(outputs)
        outputs = MaxPooling2D(pool_size=(2,2), name = 'classification_maxpool_3')(outputs)

        outputs = Convolution2D(32, (3, 3), padding='same', activation = 'relu', name = 'classification_conv_4')(outputs)
        outputs = MaxPooling2D(pool_size=(2,2), name = 'classification_maxpool_4')(outputs)

        outputs = Convolution2D(32, (3, 3), padding='same', activation = 'relu', name = 'classification_conv_5')(outputs)
        outputs = MaxPooling2D(pool_size=(2,2), name = 'classification_maxpool_5')(outputs)

        outputs = Convolution2D(32, (3, 3), padding='same', activation = 'relu', name = 'classification_conv_6')(outputs)
        outputs = MaxPooling2D(pool_size=(2,2), name = 'classification_maxpool_6')(outputs)

        outputs = Flatten()(outputs)
        outputs = Dense(256, activation = 'relu', name = 'classification_dense_1')(outputs)
        outputs = Dense(self.number_of_classes, activation='softmax', name = 'classification_dense_probs')(outputs)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])
        model.summary()

        X_in = model.input
        X_transformed = model.layers[1].output
        print(model.layers[1].name)
        self.transformation_operation = K.function([X_in], [X_transformed])

        return model

    def get_additional_callbacks(self):
        return []#[OutputsCallback(self.transformation_operation, self.preprocessor.X_train, self.preprocessor.IMG_HEIGHT, self.preprocessor.IMG_WIDTH, self.preprocessor.IMG_CHANNELS)] #return array of new callbacks [EarlyStopping(..), ..]
