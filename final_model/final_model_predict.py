import os
import time

import numpy as np
import cv2
import keras
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input


from ResNetAttentionInceptionBlocksDifferentBnorm import get_network as RNetInceptionBNorm
from ResNetAttentionInceptionBlocks import get_network as RNetInception
from ResNetAttention import get_network as RNet

def rename_layers(model, sufix):
    for i, layer in enumerate(model.layers):
        name = layer.name.split('_')[:-1]
        name = "_".join(name)
        layer.name = name + "_" + sufix + "_" + str(i)
        
def load_img(path):
    x = cv2.imread(path, 0)
    x = cv2.resize(x, (600, 1100))#, mode='constant', preserve_range=True)
    x = x.reshape((1100, 600, 1))
    return x

def get_final_prediction(predictions):
    class_votes = []
    confidence = []
    for result in res:
        class_votes.append(result[0])
        confidence.append(result[1])

    confidence = np.array(confidence)
    class_votes = np.array(class_votes)
    class_name, votes = np.unique(class_votes, return_counts = True)
    max_ind = np.argmax(votes)

    max_vote_indices = np.where(class_votes==class_name[max_ind])
    other_vote_indices = np.where(class_votes!=class_name[max_ind])
    conf_subset = confidence[max_vote_indices]
    conf_oposite_subset = confidence[other_vote_indices]

model_dict = {
    'first' : (RNet, './models/first_model.hdf5'),    
    'second' : (RNet, './models/second_model.hdf5'),    
    'third' : (RNet, './models/third_model.hdf5'),    
    'fourth' : (RNet, './models/fourth_model.hdf5'),     
    'fifth' : (RNetInception, './models/fifth_model.hdf5'), 
    'sixth' : (RNetInception, './models/sixth_model.hdf5'), 
    'seventh' : (RNetInceptionBNorm, './models/seventh_model.hdf5'),
}

inp = Input((1100,600,1))
model_outputs = []
start = time.time()

# print("=======================Building model=======================")
# for key in model_dict.keys():
#     model = model_dict[key][0]()
#     rename_layers(model, key)
#     model.load_weights(model_dict[key][1], by_name = True, skip_mismatch = True)
#     model_outputs.append(model(inp))
    
# model_ensemble = Model([inp], model_outputs)
# model_ensemble.compile(loss = ['categorical_crossentropy'] * len(model_outputs), optimizer='adam') #random loss because it is just ensembling it will not be trained further

# print("Required time to ensemble models = %s" %(time.time()-start))
# model_ensemble.summary()

from sklearn import preprocessing

classes = ['Costco',
 'Meijer',
 'HarrisTeeter',
 'KingSoopers',
 'ShopRite',
 'JewelOsco',
 'SamsClub',
 'HyVee',
 'BJs',
 'Safeway',
 'Target',
 'HEB',
 'Kroger',
 'WholeFoodsMarket',
 'StopShop',
 'FredMeyer',
 'Wegmans',
 'Walmart',
 'Frys',
 'CVSPharmacy',
 'Walgreens',
 'Publix',
 'WinCoFoods',
 'Smiths',
 'Albertsons']

le = preprocessing.LabelEncoder()
le.fit(classes)

root = '../inputs/test'
sorting_func = lambda x: int(x.split('/')[-1].split('.')[0])
threshold = 0.95 # 0.95 resnet s center lossom dao 0.901
results = []
confidence = []
original_results = []
cnt_others = 0

model_ensemble = RNetInceptionBNorm()
rename_layers(model_ensemble, 'seventh')
model_ensemble.load_weights('./models/seventh_model.hdf5', by_name = True)

for file_name in tqdm(sorted(os.listdir(root), key = sorting_func)):
    full_path = os.path.join(root, file_name)
    image = load_img(full_path)
    image = np.expand_dims(image, axis=0)
    res = np.array(model_ensemble.predict(image))
    print(res.shape)
    print(res)
    max_indices = np.argmax(res, axis=1)
    confidences = res[range(len(max_indices)),max_indices]
    print(max_indices)
    class_names = le.inverse_transform(max_indices)
    print(list(zip(class_names,confidences)))
    break