import os
import numpy as np

class BasePreprocessor():
    def __init__(self, input_directory, config_dict, name = "BasePreprocessor"):
        self.config_dict = config_dict
        self.name = name
        self.input_directory = input_directory
        self.full_path = os.path.abspath(self.input_directory)
        print("Preprocessor input path %s" % self.full_path)

        print(config_dict)
        self.IMG_WIDTH = int(config_dict['ImgWidth'])
        self.IMG_HEIGHT = int(config_dict['ImgHeight'])
        self.IMG_CHANNELS = int(config_dict['ImgChannels'])

        self.TRAIN_PATH = os.path.abspath(os.path.join(self.input_directory, 'train'))
        self.TEST_PATH = os.path.abspath(os.path.join(self.input_directory, 'test'))


        #np.random.seed(10)

    def get_shape(self):
        return (self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS)

    def get_number_of_classes(self):
        pass

    def process_data(self, x):
        pass

    def load_train(self):
        pass

    def load_validation(self):
        pass

    def load_test(self):
        pass

    def get_train(self):
        pass

    def get_validation(self):
        pass

    def get_test(self):
        pass

    def get_train_steps(self):
        pass

    def get_validation_steps(self):
        pass

    def get_test_steps(self):
        pass

    def get_train_generator(self, batch_size):
        pass

    def get_validation_generator(self, batch_size):
        pass

    def get_test_generator(self, batch_size):
        pass

