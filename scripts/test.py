import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
import re
from datetime import datetime

def sorting_key_datetime(value):
    matchObj = re.match( r'(\d{4})-(\d{2})-(\d{2})__(\d{2})_(\d{2})_(\d{2})', value, re.M|re.I)
    year = int(matchObj.group(1))
    month = int(matchObj.group(2))
    day = int(matchObj.group(3))
    hour = int(matchObj.group(4))
    minute = int(matchObj.group(5))
    second = int(matchObj.group(6))
    return datetime(year, month, day, hour, minute, second)

def sorting_key_epoch(value):
    matchObj = re.match( r'.*-(\d{4})\.hdf5', value, re.M|re.I)
    epoch = int(matchObj.group(1))
    return epoch

def get_last_model_path(preprocessor_name, model_name, checkpoints_dir):
    path = os.path.abspath(checkpoints_dir)
    path = os.path.join(path, model_name)
    path = os.path.join(path, preprocessor_name)
    last_dir = sorted(os.listdir(path), key = sorting_key_datetime, reverse = True)[0]

    path = os.path.join(path, last_dir)

    checkpoints = [x for x in os.listdir(path) if 'hdf5' in x]
    last_checkpoint = sorted(checkpoints, key = sorting_key_epoch, reverse = True)[0]

    path = os.path.join(path, last_checkpoint)
    return path

parser = argparse.ArgumentParser()
parser.add_argument("ModelName", help= "Model class name")
parser.add_argument("PreprocessorName", help = "Preprocessor class name")
parser.add_argument("-p", "--path", type=str,
                    help="Path to model checkpoint, if the argument is not given the last one will be loaded")

args = parser.parse_args()

config = configparser.ConfigParser()
config.optionxform = str
config.read("./config.cfg")

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

model_path = args.path if args.path else get_last_model_path(args.PreprocessorName, args.ModelName, checkpoint_dir)
print("LOADING MODEL %s" % model_path)
my_model.model.load_weights(model_path, by_name = True, skip_mismatch = True)

root = '../inputs/test'
root = os.path.abspath(root)
warnings.simplefilter('ignore', DeprecationWarning) #zbog sklearna i numpy deprecationa u label encoderu
warnings.filterwarnings(action='ignore')
key = lambda x: int(x.split('/')[-1].split('.')[0])

threshold = 0.95 # 0.95 resnet s center lossom dao 0.901
results = []
confidence = []
original_results = []
cnt_others = 0
for file_name in tqdm(sorted(os.listdir(root), key = key)):
    full_path = os.path.join(root, file_name)
    # image = imread(full_path)
    # image = preprocessor.process_data(image)
    image = preprocessor.process_data(full_path)
    image = np.expand_dims(image,axis=0)
    # result = my_model.model.predict(image)[0]
    result = my_model.model.predict([image, np.random.rand(1,1)])[0][0]

    max_ind = np.argmax(result)
    max_prob = result[max_ind]
    class_name = preprocessor.le.inverse_transform(max_ind)
    if max_prob < threshold:
        results.append("Other")
        cnt_others += 1 
    else:
        results.append(class_name)
    # results.append(class_name)

    original_results.append(class_name)
    confidence.append(max_prob)
sub = pd.DataFrame()
sub['Results'] = results
sub.to_csv('Mozgalo.csv', index=False, header=False)
print(len(results), " - Number of others = %d " % cnt_others) 
sub['Results'] = original_results
sub['Confidence'] = confidence
sub.to_csv('SubmissionWithConfidence.csv', index=False, header=False)
