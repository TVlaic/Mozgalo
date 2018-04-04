import os
import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Concatenate, Multiply, AveragePooling3D , Average
from keras.models import Model
from keras.optimizers import Adam

class BaseNetwork():
    def __init__(self, output_directory, checkpoint_directory, config_dict, preprocessor, name = "BaseNetwork", train = True):
        self.name = name
        self.config_dict = config_dict

        self.num_of_epochs = int(self.config_dict['Epochs'])
        self.batch_size = int(self.config_dict['BatchSize'])
        self.number_of_classes = int(self.config_dict['NumberOfClasses'])

        self.preprocessor = preprocessor

        self.output_directory = output_directory
        self.full_path = os.path.join(os.path.expanduser(self.output_directory),self.name)

        self.checkpoint_directory = checkpoint_directory
        self.root_checkpoint_path = os.path.join(os.path.expanduser(self.checkpoint_directory),self.name)

        self.curr_datetime_str = datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S")

        self.full_checkpoint_dir_path = os.path.join((self.root_checkpoint_path),self.preprocessor.name)
        self.full_checkpoint_dir_path = os.path.join(self.full_checkpoint_dir_path, self.curr_datetime_str)
        self.full_checkpoint_dir_path = os.path.expanduser(self.full_checkpoint_dir_path)

        self.tensorboard_dir_name = "tensorboard"
        self.tensorboard_path = os.path.join(self.full_checkpoint_dir_path,self.tensorboard_dir_name)

        if train:
            if not os.path.exists(self.full_path):
                os.makedirs(self.full_path)

            if not os.path.exists(self.root_checkpoint_path):
                os.makedirs(self.root_checkpoint_path)

            if not os.path.exists(self.full_checkpoint_dir_path):
                os.makedirs(self.full_checkpoint_dir_path)

            if not os.path.exists(self.tensorboard_path):
                os.makedirs(self.tensorboard_path)


            print("Imported and created %s" % self.name)
            print("Network output path %s" % self.full_path)
            print("Created root folder for network checkpoints %s" % self.root_checkpoint_path)
            print("Created network checkpoint path %s" % self.full_checkpoint_dir_path)
            print("Created tensorboard output dirs %s" %self.tensorboard_path)

    def init_network(self):
        self.model = self.get_network()

    def get_network(self):
        pass

    def get_additional_callbacks(self):
        return [] #return array of new callbacks [EarlyStopping(..), ..]

    def fit_with_generator(self):
        train_generator = self.preprocessor.get_train_generator(self.batch_size)
        validation_generator = self.preprocessor.get_validation_generator(self.batch_size)

        ckpt_name = "{val_loss:.4f}-{epoch:04d}.hdf5"
        full_ckpt_path = os.path.join(self.full_checkpoint_dir_path, ckpt_name)

        print(self.tensorboard_path)
        print(full_ckpt_path)

        self.model.fit_generator(train_generator, 
                        validation_data=validation_generator, 
                        validation_steps=self.preprocessor.get_validation_steps(), 
                        steps_per_epoch=self.preprocessor.get_train_steps(),
                        epochs=self.num_of_epochs, 
                        callbacks=[
                                    TensorBoard(log_dir=self.tensorboard_path),
                                    ModelCheckpoint(filepath=full_ckpt_path, mode = 'min', save_best_only = True, save_weights_only=True)
                                  ] + self.get_additional_callbacks(),
                        shuffle = True, #no effect if steps per epoch is not None
                        use_multiprocessing= True,
                        workers= 10,
                        max_queue_size = 10 
                        )

        pass

    def fit(self):
        curr_datetime_str = datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S")

        train_x, train_y = self.preprocessor.get_train()
        validation_x, validation_y = self.preprocessor.get_validation()

        ckpt_name = "{val_loss:.4f}-{epoch:04d}.hdf5"
        full_ckpt_path = os.path.join(self.full_checkpoint_dir_path, ckpt_name)

        print(self.tensorboard_path)
        print(full_ckpt_path)

        self.model.fit(train_x, train_y,
                       epochs=self.num_of_epochs,
                       batch_size=self.batch_size,
                       shuffle=True,
                       validation_data=(validation_x, validation_y),
                       callbacks= [
                                    TensorBoard(log_dir=self.tensorboard_path),
                                    ModelCheckpoint(filepath=full_ckpt_path, mode = 'min', save_best_only = True, save_weights_only=True)
                                  ] + self.get_additional_callbacks(),

                      )
        pass

    def eval(self):
        pass