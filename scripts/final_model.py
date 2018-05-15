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
from sklearn import preprocessing
import cv2

import warnings

import tensorflow as tf 
from keras.models import load_model
from keras.models import Model
from keras.layers import Input



def l2_embedding_loss(y_true, y_pred):
    return y_pred

def ensemble_result_prev(results_and_confidences):
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
		(votes[max_ind] >= required_number_of_votes and np.median(conf_subset) >= 0.95 and class_name[max_ind] != "Smiths" and class_name[max_ind] != "Costco") or \
		(votes[max_ind] >= required_number_of_votes-1 and len(conf_subset[conf_subset > confidence_threshold]) >= required_number_of_votes-1 and class_name[max_ind] != "Smiths" and class_name[max_ind] != "Costco") or \
		(votes[max_ind] >= len(class_votes)-3 and len(conf_subset[conf_subset > 0.9]) >= len(class_votes)-3 and class_name[max_ind] != "Smiths" and class_name[max_ind] != "Costco") or \
		(votes[max_ind] >= len(class_votes)-2 and len(conf_subset[conf_subset > 0.85]) >= len(class_votes)-2 and class_name[max_ind] != "Smiths" and class_name[max_ind] != "Costco"):  #testing this part

		return class_name[max_ind]
	else:
		return 'Other'

def process_data(x):
	x = cv2.imread(x, 0)
	x = x[:x.shape[0]//3]

	x = cv2.resize(x, (600, 1100))#, mode='constant', preserve_range=True)
	x = x.reshape((1100, 600, 1))
	return x

# parser = argparse.ArgumentParser()
# parser.add_argument("PreprocessorName", help = "Preprocessor class name")
# args = parser.parse_args()

# config = configparser.ConfigParser()
# config.optionxform = str
# config.read("./config.cfg")


# preprocessor_parameters = dict(config.items(args.PreprocessorName))
# data_parameters = dict(config.items('Data'))

# #load preprocessor
# module_name = preprocessor_parameters['PreprocessorModule']
# input_path = data_parameters['Inputs']

# module = import_module("preprocessors.%s" % (module_name))
# preprocessor_class = getattr(module, args.PreprocessorName)
# preprocessor = preprocessor_class(input_path, preprocessor_parameters) #needs to be changed for processor params or something


individual_models_folder = './IndividualModels/'
individual_models_output_folder = './IndividualModels/'
model_files = os.listdir(individual_models_folder)
model_outputs = []
model_csv_output_names = []

# inp = Input(preprocessor.get_shape())
# inp2 = Input((1,))
# for i, model_file in tqdm(enumerate(model_files), total = len(model_files)):
# 	full_path = os.path.join(individual_models_folder, model_file)
# 	model_csv_name = model_file.replace('.h5', '.csv')
# 	model_csv_output_names.append(model_csv_name)

# 	model = load_model(full_path, custom_objects={"tf": tf, 'l2_embedding_loss' : l2_embedding_loss})
# 	model.name = 'Model' + "_" +str(i)
# 	model_outputs.append(model([inp, inp2])[0])

# final_model = Model([inp,inp2], model_outputs)
# final_model.save('Final_best_model.h5')
final_model = load_model('Final_best_model.h5', custom_objects={"tf": tf, 'l2_embedding_loss' : l2_embedding_loss})

labels = ['Costco', 'Meijer', 'HarrisTeeter', 'KingSoopers', 'ShopRite', 'JewelOsco', 'SamsClub', 'HyVee', 'BJs', 'Safeway', 'Target', 'HEB', 'Kroger', 'WholeFoodsMarket', 'StopShop', 'FredMeyer', 'Wegmans', 'Walmart', 'Frys', 'CVSPharmacy', 'Walgreens', 'Publix', 'WinCoFoods', 'Smiths', 'Albertsons']
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(labels)

root = '../inputs/test'
root = os.path.abspath(root)
warnings.simplefilter('ignore', DeprecationWarning) #zbog sklearna i numpy deprecationa u label encoderu
sorting_key = lambda x: int(x.split('/')[-1].split('.')[0])

results = []
confidence = []
original_results = []
cnt_others = 0
conf_result = 0.95
for file_name in tqdm(sorted(os.listdir(root), key = sorting_key)):
	full_path = os.path.join(root, file_name)

	image = process_data(full_path)
	image = np.expand_dims(image,axis=0)


	res = np.array(final_model.predict([image, np.random.rand(1,1)]))
	res = np.squeeze(res, axis = 1)

	max_indices = np.argmax(res, axis=1)
	confidences = res[range(len(max_indices)),max_indices]
	class_names = label_encoder.inverse_transform(max_indices)
	results_and_confidences = list(zip(class_names,confidences))

	final_result = ensemble_result(results_and_confidences)
	results.append(final_result)

	original_results.append(results_and_confidences)
	

# sub = pd.DataFrame()
# sub['Results'] = results
# sub.to_csv('Mozgalo.csv', index=False, header=False)
# print(len(original_results), len(original_results[0]))
# print(list(zip(*original_results[0])))
# for i, csv_name in enumerate(model_csv_output_names):
# 	names = []
# 	confs = []
# 	for result in original_results:
# 		class_votes, confidence = list(zip(*result))
# 		names.append(class_votes[i])
# 		confs.append(confidence[i])

# 	sub = pd.DataFrame()
# 	sub['Results'] = names
# 	sub['Confidence'] = confs
# 	sub.to_csv('./IndividualResults/%s'%csv_name, index=False, header=False)
