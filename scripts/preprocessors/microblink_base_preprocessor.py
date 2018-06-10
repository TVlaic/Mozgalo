import os
import numbers

import numpy as np
from keras.utils import np_utils, generic_utils
from sklearn import preprocessing
from keras.preprocessing import image
from skimage.transform import resize
from skimage.io import imread
from sklearn.model_selection import train_test_split
from keras.utils import Sequence
from .base_preprocessor import BasePreprocessor

import cv2

class GeneratorWrapper(Sequence):
    def __init__(self, x, y, curiculum_epochs, batch_size, image_shape, keyword_args, preprocess_data_func):
        self.x, self.y = x, y
        self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS = image_shape
        self.batch_size = batch_size
        self.keyword_args = keyword_args
        self.epoch_num = 0
        self.preprocess_data = preprocess_data_func
        self.curiculum_epochs = curiculum_epochs
        self.datagen = self.create_datagen()


    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        X1 = np.zeros((self.batch_size, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        Y = np.zeros((self.batch_size, batch_y.shape[1]))

        for i, ind in enumerate(batch_x):
            image = self.preprocess_data(batch_x[i])

            X1[i] = self.datagen.random_transform(image)
            Y[i] = batch_y[i]
        return X1, Y

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        self.epoch_num +=1
        self.datagen = self.create_datagen()

    def create_datagen(self):
        if len(self.keyword_args.keys()) == 0:
            return image.ImageDataGenerator()
        else:
            if self.curiculum_epochs == 0:
                weight = 1
            else:
                weight = min(1, self.epoch_num/self.curiculum_epochs)
            new_datagen = {}
            for key in self.keyword_args:
                if isinstance(self.keyword_args[key], numbers.Number):
                    new_datagen[key] = self.keyword_args[key] * weight
                else:
                    new_datagen[key] = self.keyword_args[key]

            return image.ImageDataGenerator(**new_datagen)

class MicroblinkBasePreprocessor(BasePreprocessor):
    def __init__(self, input_directory, config_dict, name = "MicroblinkBasePreprocessor"):
        BasePreprocessor.__init__(self, input_directory, config_dict, name = name)
        self.top_side_only = "true" == config_dict['TopSideOnly'].lower()

        self.TEST_PERCENTAGE = float(config_dict['TestPercentage'])

        #augmentations
        self.width_shift_range = float(config_dict['WidthShiftRange']) if 'WidthShiftRange' in config_dict else 0.
        self.height_shift_range = float(config_dict['HeightShiftRange']) if 'HeightShiftRange' in config_dict  else 0.
        self.shear_range = float(config_dict['ShearRange']) if 'ShearRange' in config_dict  else 0.
        self.rotation_range = int(config_dict['RotationRange']) if 'RotationRange' in config_dict  else 0
        self.curiculum_epochs = int(config_dict['CuriculumEpochs']) if 'CuriculumEpochs' in config_dict else 0

        self.le = preprocessing.LabelEncoder()
        self.X_train = []
        self.y_train = []
        for root, dirs, files in os.walk(self.TRAIN_PATH, topdown=False):
            for name in files:
                self.X_train.append(os.path.join(root,name))
                self.y_train.append(root.split('/')[-1])

        self.le.fit(self.y_train)
        self.y_train = self.le.transform(self.y_train)

        self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(self.X_train, self.y_train, test_size=self.TEST_PERCENTAGE, random_state=42, stratify=self.y_train)

        self.X_train = np.array(self.X_train)
        self.X_validation = np.array(self.X_validation)
        self.y_train = np_utils.to_categorical(self.y_train, len(set(self.le.classes_)))
        self.y_validation = np_utils.to_categorical(self.y_validation, len(set(self.le.classes_)))


    def get_number_of_classes(self):
        return len(os.listdir(self.TRAIN_PATH))

    def get_train_steps(self):
        return None #because i got the sequential wrapper for generating data  self.X_train.shape[0]/self.TRAIN_BATCH_SIZE

    def get_validation_steps(self):
        return self.X_validation.shape[0]/self.VALIDATION_BATCH_SIZE

    def get_test_steps(self):
        pass

    def process_data(self, x):
        if self.IMG_CHANNELS==1:
            x = cv2.imread(x, 0)
        else:
            x = cv2.imread(x)

        if self.top_side_only:
            x = x[:x.shape[0]//3]

        x = cv2.resize(x, (self.IMG_WIDTH, self.IMG_HEIGHT))#, mode='constant', preserve_range=True)
        if len(x.shape) != 3 and self.IMG_CHANNELS == 3:
            x = np.dstack([x,x,x])
        elif self.IMG_CHANNELS == 1:
            x = x.reshape((self.IMG_HEIGHT, self.IMG_WIDTH, 1))
        return x

    def generator_wrapper(self, x, y, datagen, batch_size, shuffle = True):
        while True:
            if shuffle:
                indices = np.random.permutation(np.arange(y.shape[0]))
            else:
                indices = np.arange(y.shape[0])
            num_batches = y.shape[0] // batch_size
            assert y.shape[0] % batch_size == 0

            for bid in range(num_batches):
                batch_indices = indices[bid * batch_size : (bid + 1) * batch_size]
                X1 = np.zeros((batch_size, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
                Y = np.zeros((batch_size, y.shape[1]))

                for i, ind in enumerate(batch_indices):
                    image = self.process_data(imread(x[ind]))
                    X1[i] = datagen.random_transform(image)
                    Y[i] = y[ind]

                yield X1, Y

    def get_train_generator(self, batch_size):  
        self.TRAIN_BATCH_SIZE = batch_size      
        datagen_args = dict(width_shift_range = self.width_shift_range,
                            height_shift_range = self.height_shift_range,
                            shear_range = self.shear_range, 
                            rotation_range = self.rotation_range)

        return GeneratorWrapper(self.X_train, self.y_train, self.curiculum_epochs, batch_size, (self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), datagen_args, self.process_data)

    def get_validation_generator(self, batch_size): 
        self.VALIDATION_BATCH_SIZE = batch_size
        
        return GeneratorWrapper(self.X_validation, self.y_validation, self.curiculum_epochs, batch_size, (self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), {}, self.process_data)

    def get_test_generator(self, batch_size):
        pass

    def load_train(self):
        raise Exception("Not implemented")

    def load_validation(self):
        raise Exception("Not implemented")

    def load_test(self):
        raise Exception("Not implemented")

    def get_train(self):
        raise Exception("Not implemented")

    def get_validation(self):
        raise Exception("Not implemented")

    def get_test(self):
        raise Exception("Not implemented")

