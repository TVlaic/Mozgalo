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


def get_img_aug(keyword_args):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    if len(keyword_args.keys()) != 0:
        seq = iaa.Sequential(
        [
            sometimes(iaa.Affine(
                scale = {"x": keyword_args['scale_x'], "y": keyword_args['scale_y']}, # scale images to 80-120% of their size, individually per axis
                translate_percent = {"x": keyword_args['translate_x'], "y": keyword_args['translate_y']}, # translate by -10 to +10 percent (per axis)
                rotate = keyword_args['rotation_range'], # rotate by -15 to +15 degrees
                shear = keyword_args['shear_range'],
                cval = (0, 255), # if mode is constant, use a cval between 0 and 255
                mode = "constant" # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 2),
                [
                    iaa.OneOf([
                        iaa.GaussianBlur(keyword_args['gaussian_blur_range']), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k = keyword_args['average_blur_kernel']), # blur image using local means with kernel sizes between 2 and 5
                        iaa.MedianBlur(k = keyword_args['median_blur_kernel']), # blur image using local medians with kernel sizes between 2 and 5
                    ]),
                    iaa.AdditiveGaussianNoise(loc = 0, scale = keyword_args['additive_gaussian_noise_scale'], per_channel = 0.5),
                    iaa.Dropout(keyword_args['dropout_range'], per_channel = 0.5),
                    iaa.Add(keyword_args['add_range'], per_channel = 0.5), # change brightness of images (by -10 to 10 of original value)
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.Multiply(keyword_args['multiply_range'], per_channel = 0.5),
                    iaa.ElasticTransformation(alpha = keyword_args['elastic_transform_alpha_range'], sigma = keyword_args['elastic_transform_sigma']),
                    iaa.PerspectiveTransform(scale = keyword_args['perspective_transform_scale'])

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
    else:
        seq = iaa.Sequential(iaa.Noop())
    return seq

class GeneratorWrapper(Sequence):
    def __init__(self, x, y, batch_size, image_shape, keyword_args, preprocess_data_func):
        self.x, self.y = x, y
        self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS = image_shape
        self.batch_size = batch_size
        self.keyword_args = keyword_args
        self.epoch_num = 0
        self.preprocess_data = preprocess_data_func
        self.seq = get_img_aug(keyword_args)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        X1 = np.zeros((self.batch_size, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS)).astype(np.uint8)
        Y = np.zeros((self.batch_size, batch_y.shape[1]))

        for i, ind in enumerate(batch_x):
            image = self.preprocess_data(batch_x[i])
            X1[i] = image
            Y[i] = batch_y[i]

        if len(self.keyword_args) > 0:
            X1 = self.seq.augment_images(X1)
        return [X1, np.argmax(Y, axis=1)], [Y, np.random.rand(self.batch_size,1)]



def get_keyword_args(config_dict):

    scale_range_width = parse_range(config_dict['ScaleRangeX']) if 'ScaleRangeX' in config_dict else (1.,1.)
    scale_range_height = parse_range(config_dict['ScaleRangeY']) if 'ScaleRangeY' in config_dict else (1.,1.)
    width_shift_range = parse_range(config_dict['WidthShiftRange']) if 'WidthShiftRange' in config_dict else (0.,0.)
    height_shift_range = parse_range(config_dict['HeightShiftRange']) if 'HeightShiftRange' in config_dict  else (0.,0.)
    shear_range = parse_range(config_dict['ShearRange']) if 'ShearRange' in config_dict  else (0.,0.)
    rotation_range = parse_range(config_dict['RotationRange']) if 'RotationRange' in config_dict  else (0.,0.)
    gaussian_blur_range = parse_range(config_dict['GaussianBlurSigma']) if 'GaussianBlurSigma' in config_dict  else (0.,0.)
    median_blur_kernel = parse_range(config_dict['MedianBlurKernel']) if 'MedianBlurKernel' in config_dict  else (0.,0.)
    average_blur_kernel = parse_range(config_dict['AverageBlurKernel']) if 'AverageBlurKernel' in config_dict  else (0.,0.)
    additive_gaussian_noise_scale = parse_range(config_dict['AdditiveGaussianNoiseScale']) if 'AdditiveGaussianNoiseScale' in config_dict  else (0.,0.)
    dropout_range = parse_range(config_dict['Dropout']) if 'Dropout' in config_dict  else (0.,0.)
    add_range = parse_range(config_dict['Add']) if 'Add' in config_dict  else (0.,0.)
    multiply_range = parse_range(config_dict['Multiply']) if 'Multiply' in config_dict  else (0.,0.)
    elastic_transform_alpha_range = parse_range(config_dict['ElasticTransformationAlpha']) if 'ElasticTransformationAlpha' in config_dict  else (0.,0.)
    elastic_transform_sigma = float(config_dict['ElasticTransformationSigma']) if 'ElasticTransformationSigma' in config_dict else 0.2
    perspective_transform_scale = parse_range(config_dict['PerspectiveTransformScale']) if 'PerspectiveTransformScale' in config_dict  else (0.,0.)

    return dict(scale_x = scale_range_width,
                scale_y = scale_range_height,
                translate_x = width_shift_range,
                translate_y = height_shift_range,
                shear_range = shear_range, 
                rotation_range = rotation_range,
                gaussian_blur_range = gaussian_blur_range,
                median_blur_kernel = median_blur_kernel,
                average_blur_kernel = average_blur_kernel,
                additive_gaussian_noise_scale = additive_gaussian_noise_scale,
                dropout_range = dropout_range,
                add_range = add_range,
                multiply_range = multiply_range,
                elastic_transform_alpha_range = elastic_transform_alpha_range,
                elastic_transform_sigma = elastic_transform_sigma,
                perspective_transform_scale = perspective_transform_scale)


def parse_range(line):
    vals = line.split(',')
    x1 = float(vals[0].strip())
    x2 = float(vals[1].strip())
    return (x1,x2)


class MicroblinkBasePreprocessorImgaugCenterLoss(BasePreprocessor):
    def __init__(self, input_directory, config_dict, name = "MicroblinkBasePreprocessorImgaugCenterLoss"):
        BasePreprocessor.__init__(self, input_directory, config_dict, name = name)
        self.top_side_only = "true" == config_dict['TopSideOnly'].lower()

        self.TEST_PERCENTAGE = float(config_dict['TestPercentage'])

        self.keyword_args = get_keyword_args(config_dict)

        print('Augmentation parameters')
        for key in self.keyword_args.keys():
            print(key, ' - ', self.keyword_args[key])
            
        #augmentations
        # self.width_shift_range = parse_range(config_dict['WidthShiftRange']) if 'WidthShiftRange' in config_dict else (0.,0.)
        # self.height_shift_range = parse_range(config_dict['HeightShiftRange']) if 'HeightShiftRange' in config_dict  else (0.,0.)
        # self.shear_range = parse_range(config_dict['ShearRange']) if 'ShearRange' in config_dict  else (0.,0.)
        # self.scale_range = parse_range(config_dict['ScaleRange']) if 'ScaleRange' in config_dict  else (0.,0.)
        # self.rotation_range = parse_range(config_dict['RotationRange']) if 'RotationRange' in config_dict  else (0.,0.)
        # self.gaussian_blur_range = parse_range(config_dict['GaussianBlur']) if 'GaussianBlur' in config_dict  else (0.,0.)
        # self.median_blur_range = parse_range(config_dict['MedianBlur']) if 'MedianBlur' in config_dict  else (0.,0.)
        # self.average_blur_range = parse_range(config_dict['AverageBlur']) if 'AverageBlur' in config_dict  else (0.,0.)
        # self.additive_gaussian_noise_range = parse_range(config_dict['AdditiveGaussianNoise']) if 'AdditiveGaussianNoise' in config_dict  else (0.,0.)
        # self.dropout_range = parse_range(config_dict['Dropout']) if 'Dropout' in config_dict  else (0.,0.)
        # self.add_range = parse_range(config_dict['Add']) if 'Add' in config_dict  else (0.,0.)
        # self.multiply_range = parse_range(config_dict['Multiply']) if 'Multiply' in config_dict  else (0.,0.)
        # self.elastic_transform_range = parse_range(config_dict['ElasticTransformation']) if 'ElasticTransformation' in config_dict  else (0.,0.)
        # self.perspective_transform_range = parse_range(config_dict['PerspectiveTransform']) if 'PerspectiveTransform' in config_dict  else (0.,0.)
        # self.elastic_transform_sigma = float(config_dict['ElasticTransformationSigma']) if 'ElasticTransformationSigma' in config_dict else 0.2

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
        return GeneratorWrapper(self.X_train, self.y_train, batch_size, (self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), self.keyword_args, self.process_data)

    def get_validation_generator(self, batch_size): 
        self.VALIDATION_BATCH_SIZE = batch_size

        return GeneratorWrapper(self.X_validation, self.y_validation, batch_size, (self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), {}, self.process_data)

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

