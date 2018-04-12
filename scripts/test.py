import os
import argparse
import configparser
from shutil import copyfile
from importlib import import_module

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


my_model.model.load_weights('/home/user/Mozgalo/checkpoints/ResidualAttentionNet/MicroblinkBasePreprocessorWithFakes/2018-04-12__12_35_19/0.0256-0001.hdf5', by_name = True, skip_mismatch = True)
# raise Exception("definiraj model")
root = '../inputs/test'
root = os.path.abspath(root)
warnings.simplefilter('ignore', DeprecationWarning) #zbog sklearna i numpy deprecationa u label encoderu
key = lambda x: int(x.split('/')[-1].split('.')[0])
threshold = 0.95 #dalo 0.68  rezultat

results = []
for file_name in tqdm(sorted(os.listdir(root), key = key)):
    full_path = os.path.join(root, file_name)
    # image = imread(full_path)
    # image = preprocessor.process_data(image)
    image = preprocessor.process_data(full_path)
    image = np.expand_dims(image,axis=0)
    result = my_model.model.predict(image)[0]

    max_ind = np.argmax(result)
    max_prob = result[max_ind]
    if max_prob < threshold:
        results.append("Other")
    else:
        class_name = preprocessor.le.inverse_transform(max_ind)
        results.append(class_name)


sub = pd.DataFrame()
sub['Results'] = results
sub.to_csv('Mozgalo.csv', index=False, header=False)
print(len(results))