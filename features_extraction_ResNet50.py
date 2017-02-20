from __future__ import print_function 
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import Input, Dense
from sklearn.preprocessing import normalize
import os
import sys
import h5py
from sklearn.metrics.pairwise import cosine_similarity
import scipy.spatial.distance

img_path = '/imatge/mcompri/Desktop/THESIS/Places-CNN/ESEMPIO_coffe2keras/keras-master/keras/caffe/modello_prova/test_jpeg/images_test/'

ds = open('dataset_new.txt', 'w')

# LOAD MODEL
json_file = open('ResNet50_prova.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# LOAD WEIGHTS

loaded_model.load_weights("resnet50_weights_prova.h5")

# CHOOSE LAYER TO EXTRACT FEATURES VECTORS

#base_model = ResNet50(weights='imagenet')#, include_top=True)

model = Model(input=loaded_model.input, output=loaded_model.get_layer('avg_pool').output)  # sotituisci base_ con loaded_



# LOAD TEST IMAGES

inputs = Input(shape=(3,224,224,))
#print(loaded_model.input.shape)

f = h5py.File('resnet50_cs_prova.h5', 'w')

matrix = []

im_array = []
a = os.popen( 'ls /imatge/mcompri/Desktop/THESIS/Places-CNN/ESEMPIO_coffe2keras/keras-master/keras/caffe/modello_prova/test_jpeg/images_test/').read()	
print (a.split('\n')) 
for g in a.split('\n')[:-1]:
	b = os.popen('ls /imatge/mcompri/Desktop/THESIS/Places-CNN/ESEMPIO_coffe2keras/keras-master/keras/caffe/modello_prova/test_jpeg/images_test/'+g).read() #test_prova = 1680 images  , test = 400 images
	
	for p,i in enumerate(b.split('\n')[:-1]):
	
			if '.tif' in i:    
				
					
				img_path_1 = img_path +g +'/'+i 
				img = image.load_img(img_path_1, target_size=(224, 224))
				x = image.img_to_array(img)
				x = np.expand_dims(x, axis=0)
				x = preprocess_input(x)
				features = model.predict(x)  #(1, 2048, 1, 1)
				features = features[0] # (2048, 1, 1)
				features = np.array(features)
				print(features.shape)
				feat_vec = np.empty(2048)
				results = np.empty(420)
				for c in range(len(features)):
					for d in range (len(features[0][0])):
						feat_vec[c] = np.amax(features[c][d])
				# features normalization		
				norm = np.linalg.norm(feat_vec)
				feat_vec = [float(i)/norm for i in feat_vec]
				matrix.append(feat_vec)

# cosine similarity computation
						
matrix_new = []

for j in matrix :
	row = []
	j = np.array(j).reshape(1,-1)
	#print (i)                                
	for i in matrix :
		i = np.array(i).reshape(1,-1)
		row.append(cosine_similarity(j,i)[0][0])
		#print(row)
	matrix_new.append(row)
#print(np.array(matrix_new))
f.create_dataset('cosine_similarity_resnet50_prova' , data=matrix_new) #file containing feature vector normalized				
