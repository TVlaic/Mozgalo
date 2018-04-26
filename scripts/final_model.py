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

import tensorflow as tf 
from keras.models import load_model
from keras.models import Model
from keras.layers import Input



def l2_embedding_loss(y_true, y_pred):
    return y_pred

def ensemble_result(results_and_confidences):
	confidence_threshold = 0.95 
	class_votes, confidence = list(zip(*results_and_confidences))

	confidence = np.array(confidence)
	class_votes = np.array(class_votes)
	class_name, votes = np.unique(class_votes, return_counts = True)
	max_ind = np.argmax(votes)

	max_vote_indices = np.where(class_votes==class_name[max_ind])
	other_vote_indices = np.where(class_votes!=class_name[max_ind])
	conf_subset = confidence[max_vote_indices]
	conf_oposite_subset = confidence[other_vote_indices]

	required_number_of_votes = np.ceil(len(class_votes)/2)
	if (votes[max_ind] >= required_number_of_votes and len(conf_subset[conf_subset > confidence_threshold]) >= required_number_of_votes) or \
		(votes[max_ind] >= required_number_of_votes-1 and len(conf_subset[conf_subset > confidence_threshold]) >= required_number_of_votes-1 and class_name[max_ind] != "Smiths" and class_name[max_ind] != "Costco") or \
		(votes[max_ind] >= len(class_votes)-3 and len(conf_subset[conf_subset > 0.9]) >= len(class_votes)-3 and class_name[max_ind] != "Smiths" and class_name[max_ind] != "Costco") or \
		(votes[max_ind] >= len(class_votes)-2 and len(conf_subset[conf_subset > 0.85]) >= len(class_votes)-2 and class_name[max_ind] != "Smiths" and class_name[max_ind] != "Costco"):  #testing this part

		return class_name[max_ind]
	else:
		return 'Other'

parser = argparse.ArgumentParser()
parser.add_argument("PreprocessorName", help = "Preprocessor class name")
args = parser.parse_args()

config = configparser.ConfigParser()
config.optionxform = str
config.read("./config.cfg")


preprocessor_parameters = dict(config.items(args.PreprocessorName))
data_parameters = dict(config.items('Data'))

#load preprocessor
module_name = preprocessor_parameters['PreprocessorModule']
input_path = data_parameters['Inputs']

module = import_module("preprocessors.%s" % (module_name))
preprocessor_class = getattr(module, args.PreprocessorName)
preprocessor = preprocessor_class(input_path, preprocessor_parameters) #needs to be changed for processor params or something


individual_models_folder = './IndividualModels/'
individual_models_output_folder = './IndividualModels/'
model_files = os.listdir(individual_models_folder)
model_outputs = []
model_csv_output_names = []

image = preprocessor.process_data('/home/user/Mozgalo/inputs/test/0.jpg')
image = np.expand_dims(image,axis=0)


inp = Input(preprocessor.get_shape())
inp2 = Input((1,))
for i, model_file in tqdm(enumerate(model_files), total = len(model_files)):
	full_path = os.path.join(individual_models_folder, model_file)
	model_csv_name = model_file.replace('.h5', '.csv')
	model_csv_output_names.append(model_csv_name)
	model = load_model(full_path, custom_objects={"tf": tf, 'l2_embedding_loss' : l2_embedding_loss})
	model.name = 'Model' + "_" +str(i)
	model_outputs.append(model([inp, inp2])[0])

final_model = Model([inp,inp2], model_outputs)

root = '../inputs/test'
root = os.path.abspath(root)
warnings.simplefilter('ignore', DeprecationWarning) #zbog sklearna i numpy deprecationa u label encoderu
sorting_key = lambda x: int(x.split('/')[-1].split('.')[0])

results = []
confidence = []
original_results = []
cnt_others = 0
for file_name in tqdm(sorted(os.listdir(root), key = sorting_key)):
	full_path = os.path.join(root, file_name)

	image = preprocessor.process_data(full_path)
	image = np.expand_dims(image,axis=0)


	res = np.array(final_model.predict([image, np.random.rand(1,1)]))
	res = np.squeeze(res, axis = 1)

	max_indices = np.argmax(res, axis=1)
	confidences = res[range(len(max_indices)),max_indices]
	class_names = preprocessor.le.inverse_transform(max_indices)
	results_and_confidences = list(zip(class_names,confidences))
	original_results.append(results_and_confidences)

	final_result = ensemble_result(results_and_confidences)
	results.append(final_result)


	original_results.append(results_and_confidences)

sub = pd.DataFrame()
sub['Results'] = results
sub.to_csv('Mozgalo.csv', index=False, header=False)
