import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import configparser
from shutil import copyfile
from importlib import import_module
import keras.backend as K
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from tqdm import tqdm

from PIL import ImageFile #jedan je buggy malo
ImageFile.LOAD_TRUNCATED_IMAGES = True 
from skimage.io import imread

import warnings

parser = argparse.ArgumentParser()
parser.add_argument("ModelName", help= "Model class name")
parser.add_argument("PreprocessorName", help = "Preprocessor class name")
args = parser.parse_args()

config = configparser.ConfigParser()
config.optionxform = str
config.read("./config.cfg")

'''
for sect in config.sections():
    section_items = dict(config.items(sect))
    print("Section %s" % sect)
    for key in section_items.keys():
        print("Key: %s" % section_items[key])
''' and None

model_parameters = dict(config.items(args.ModelName))
preprocessor_parameters = dict(config.items(args.PreprocessorName))
data_parameters = dict(config.items('Data'))

#load preprocessor
module_name = preprocessor_parameters['PreprocessorModule']
input_path = data_parameters['Inputs']

module = import_module("preprocessors.%s" % (module_name))
preprocessor_class = getattr(module, args.PreprocessorName)
preprocessor = preprocessor_class(input_path, preprocessor_parameters) #needs to be changed for processor params or something

#load network model
module_name = model_parameters['ModelModule']
checkpoint_dir = data_parameters['CheckpointDirectory']
result_path = data_parameters['Outputs']

module = import_module("models.%s" % (module_name))
ml_class= getattr(module, args.ModelName)
my_model = ml_class(result_path, checkpoint_dir, model_parameters, preprocessor, train=False)
my_model.init_network()


my_model.model.load_weights('/home/user/Mozgalo/checkpoints/ResidualAttentionNetSmall/MicroblinkBasePreprocessorWithFakes/2018-04-14__18_25_35/0.0128-0045.hdf5', by_name = True, skip_mismatch = True)
# my_model.model.load_weights('/home/user/Mozgalo/checkpoints/ResidualAttentionNetSmallDifferentInterpolationCenterLoss/MicroblinkBasePreprocessorImgaugCenterLoss/2018-04-18__18_22_29_0.901/0.0341-0011.hdf5', by_name = True, skip_mismatch = True)
# raise Exception("definiraj model")
root = '../inputs/test'
root = os.path.abspath(root)
warnings.simplefilter('ignore', DeprecationWarning) #zbog sklearna i numpy deprecationa u label encoderu
warnings.filterwarnings(action='ignore')
key = lambda x: int(x.split('/')[-1].split('.')[0])

# print(my_model.model.summary())
# print(my_model.model.input)
# print(my_model.model.output)
# print(my_model.model.get_layer('classification_dense_1'))
# print(K.Function([my_model.model.input[0]], [my_model.model.get_layer('classification_dense_1').output]))
get_embed = K.Function([my_model.model.input], [my_model.model.get_layer('classification_dense_1').output])


X_train, _, y_train, _ = train_test_split(preprocessor.X_train, preprocessor.y_train, test_size=0.9, random_state=42, stratify=preprocessor.y_train)
embeds = []
embed_classes = []

threshold = 0.95 # 0.95 resnet s center lossom dao 0.901
results = []
confidence = []
original_results = []
cnt_others = 0
for i, file_name in tqdm(enumerate(X_train), total=len(list(X_train))):
    full_path = os.path.join(root, file_name)
    # image = imread(full_path)
    # image = preprocessor.process_data(image)
    image = preprocessor.process_data(full_path)
    image = np.expand_dims(image,axis=0)
    # result = my_model.model.predict(image)[0]

    embeds.append(get_embed([image]))
    embed_classes.append(preprocessor.le.inverse_transform(np.argmax(y_train[i])))

embeds = np.array(embeds)
embed_classes = np.array(embed_classes)

np.save('embeds2.npy', embeds)
np.save('embed_classes2.npy', embed_classes)