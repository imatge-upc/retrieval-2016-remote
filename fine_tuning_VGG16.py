from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from keras import backend as K
from keras.optimizers import SGD, Adam, rmsprop
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Merge, merge
from keras.layers import Activation, Dropout, Flatten, Dense 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from mlxtend.preprocessing import one_hot


# link da guardare per fine-tuning--> https://github.com/fchollet/keras/issues/871

img_path = '/imatge/mcompri/Desktop/THESIS/Places-CNN/ESEMPIO_coffe2keras/keras-master/keras/caffe/modello_prova/training/images_training/'

# create the base pre-trained model
base_model = VGG16(weights='imagenet', include_top=False)
'''
###     base_model Architecture

		(0, 'input_1')
		(1, 'block1_conv1')
		(2, 'block1_conv2')
		(3, 'block1_pool')
		(4, 'block2_conv1')
		(5, 'block2_conv2')
		(6, 'block2_pool')
		(7, 'block3_conv1')
		(8, 'block3_conv2')
		(9, 'block3_conv3')
		(10, 'block3_pool')
		(11, 'block4_conv1')
		(12, 'block4_conv2')
		(13, 'block4_conv3')
		(14, 'block4_pool')
		(15, 'block5_conv1')
		(16, 'block5_conv2')
		(17, 'block5_conv3')
		(18, 'block5_pool')
'''

for i, layer in enumerate(base_model.layers):
   print(i, layer.name)


layer_name = 'block5_pool'
intermediate_layer_model = Model(input=base_model.input,output=base_model.get_layer(layer_name).output)
#intermediate_output = intermediate_layer_model.predict(data)
x = base_model.get_layer(layer_name).output
Dropout(0.3)
x = GlobalAveragePooling2D()(x)
fc1 = Dense(1000, activation='tanh', name = 'fc1')(x)
b1=BatchNormalization(axis=1)(fc1)
fc2 = Dense(1000,activation='tanh', name = 'fc2')(b1)
b2=BatchNormalization(axis=1)(fc2)
predictions = Dense(17,activation = 'sigmoid')(b2) #



# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional VGG16 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='binary_crossentropy')

# IMAGES LOAD

im_array = []
a = os.popen( 'ls /imatge/mcompri/Desktop/THESIS/Places-CNN/ESEMPIO_coffe2keras/keras-master/keras/caffe/modello_prova/training/images_training/').read()	
print (a.split('\n')) 
for g in a.split('\n')[:-1]:
	b = os.popen('ls /imatge/mcompri/Desktop/THESIS/Places-CNN/ESEMPIO_coffe2keras/keras-master/keras/caffe/modello_prova/training/images_training/'+g).read() #test_prova = 1680 images  , test = 400 images

		#b = os.popen( 'ls '+img_path).read()
	for p,i in enumerate(b.split('\n')[:-1]):
		#for i in b.split('\n')[:-1]:
			if '.tif' in i:    
				
				
				img_path_1 = img_path +g +'/'+i 
				img = image.load_img(img_path_1, target_size=(224, 224))
				name=os.path.basename(img_path_1)
				img = image.img_to_array(img)
				im_array.append(img)
								
im_array = np.array(im_array)
trainFeatures = im_array  # (1680,3,224,224


# LABELS LOAD


dataframe = pd.read_csv("multi_label_training_1680.csv", header=None)
dataset = dataframe.values
r = np.array(dataset)  # (1680,17)



# train the model on the new data for a few epochs
#model.fit_generator(...)
model.fit(trainFeatures,r,batch_size=32,nb_epoch=10,verbose=2)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:10]:
   layer.trainable = False
for layer in model.layers[10:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0000001), loss='binary_crossentropy') #categorical_crossentropy
#model.compile(optimizer=Adam(lr=0.0000001), loss='categorical_crossentropy')
#model.compile(optimizer=Adadelta(lr=1.0, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
#model.fit_generator(...)
model.fit(trainFeatures,r,batch_size=32,validation_split=0.4,nb_epoch=10,verbose=2) # from 32 to 40 to avoid large GPU memeory cost (from cines herbal paper) 
predictions = model.predict(trainFeatures)
#rounded = [round(x) for x in predictions]
print(predictions)
print("saving model and weights")
model_json = model.to_json()
with open("vgg16_prova.json", "w") as json_file:
	json_file.write(model_json)
print("saving...")
model.save_weights('vgg16_weights_prova.h5')
