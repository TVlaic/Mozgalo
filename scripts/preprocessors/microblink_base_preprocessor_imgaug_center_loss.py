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

import imgaug as ia
from imgaug import augmenters as iaa
import cv2

def get_img_aug():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
    [
        sometimes(iaa.Affine(
            #nisam siguran za ovaj scaling tho
            scale={"x": (0.85, 1.0), "y": (0.85, 1.0)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.1, 0.1), "y": (0., 0.)}, # translate by -10 to +10 percent (per axis)
            rotate=(-15, 15), # rotate by -15 to +15 degrees
            rotate=(-20, 20), # rotate by -15 to +15 degrees
            # shear=(-15, 15), # shear by -16 to +16 degrees
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode="constant" # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 2),
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 5)), # blur image using local means with kernel sizes between 2 and 5
                    iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 5
                ]),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=0.5),
                iaa.Dropout((0.01, 0.1), per_channel=0.5),
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                iaa.ElasticTransformation(alpha=(0.5, 2.), sigma=0.2),
                iaa.PerspectiveTransform(scale=(0.01, 0.075))
                #only works for colored images
                #iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                #iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                #iaa.Grayscale(alpha=(0.0, 1.0)),
            ],
            random_order=True
        )
    ],
    random_order=True
    )
    return seq

class GeneratorWrapper(Sequence):
    def __init__(self, x, y, curiculum_epochs, batch_size, image_shape, keyword_args, preprocess_data_func):
        self.x, self.y = x, y
        # self.datagen = datagen
        self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS = image_shape
        self.batch_size = batch_size
        self.keyword_args = keyword_args
        self.epoch_num = 0
        self.preprocess_data = preprocess_data_func
        self.curiculum_epochs = curiculum_epochs
        self.datagen = self.create_datagen()
        self.seq = get_img_aug()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        X1 = np.zeros((self.batch_size, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS)).astype(np.uint8)
        Y = np.zeros((self.batch_size, batch_y.shape[1]))

        cut_off_fake_examples = 0.3
        flip = False
        # if len(self.keyword_args) > 0:
        #     flip = True
        # flip=True
        for i, ind in enumerate(batch_x):
            image = self.preprocess_data(batch_x[i])
            X1[i] = image

            if flip and np.random.rand() < 1./20: # small residual net treniran ovako 1./15:
                X1[i] = np.fliplr(X1[i])
                Y[i] = np.zeros((1, batch_y.shape[1])) + 1./batch_y.shape[1]
                continue
            Y[i] = batch_y[i]

        if len(self.keyword_args) > 0:
            X1 = self.seq.augment_images(X1)
        return [X1, np.argmax(Y, axis=1)], [Y, np.random.rand(self.batch_size,1)]

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

class MicroblinkBasePreprocessorImgaugCenterLoss(BasePreprocessor):
    def __init__(self, input_directory, config_dict, name = "MicroblinkBasePreprocessorImgaugCenterLoss"):
        BasePreprocessor.__init__(self, input_directory, config_dict, name = name)
        self.top_side_only = bool(config_dict['TopSideOnly'])

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
        self.y_train = np_utils.to_categorical(self.y_train, len(set(self.le.classes_)))

        self.X_validation = np.array(self.X_validation)
        self.y_validation = np_utils.to_categorical(self.y_validation, len(set(self.le.classes_)))

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

        # image_datagen = image.ImageDataGenerator(**datagen_args)
        # image_datagen = image.ImageDataGenerator()

        # return self.generator_wrapper(self.X_train, self.y_train, image_datagen, batch_size)
        return GeneratorWrapper(self.X_train, self.y_train, self.curiculum_epochs, batch_size, (self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), datagen_args, self.process_data)

    def get_validation_generator(self, batch_size): 
        self.VALIDATION_BATCH_SIZE = batch_size

        # image_datagen = image.ImageDataGenerator()

        # return self.generator_wrapper(self.X_validation, self.y_validation, image_datagen, batch_size, shuffle=False)
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

