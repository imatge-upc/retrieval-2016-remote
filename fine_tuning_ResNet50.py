from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from mlxtend.preprocessing import one_hot
import matplotlib
#matplotlib.use('pdf')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# link da guardare per fine-tuning--> https://github.com/fchollet/keras/issues/871

# create the base pre-trained model
base_model = ResNet50(weights='imagenet')#, include_top=False)

###     base_model Architecture
'''
(0, 'input_1')
(1, 'zeropadding2d_1')
(2, 'conv1')
(3, 'bn_conv1')
(4, 'activation_1')
(5, 'maxpooling2d_1')
(6, 'res2a_branch2a')
(7, 'bn2a_branch2a')
(8, 'activation_2')
(9, 'res2a_branch2b')
(10, 'bn2a_branch2b')
(11, 'activation_3')
(12, 'res2a_branch2c')
(13, 'res2a_branch1')
(14, 'bn2a_branch2c')
(15, 'bn2a_branch1')
(16, 'merge_1')
(17, 'activation_4')
(18, 'res2b_branch2a')
(19, 'bn2b_branch2a')
(20, 'activation_5')
(21, 'res2b_branch2b')
(22, 'bn2b_branch2b')
(23, 'activation_6')
(24, 'res2b_branch2c')
(25, 'bn2b_branch2c')
(26, 'merge_2')
(27, 'activation_7')
(28, 'res2c_branch2a')
(29, 'bn2c_branch2a')
(30, 'activation_8')
(31, 'res2c_branch2b')
(32, 'bn2c_branch2b')
(33, 'activation_9')
(34, 'res2c_branch2c')
(35, 'bn2c_branch2c')
(36, 'merge_3')
(37, 'activation_10')
(38, 'res3a_branch2a')
(39, 'bn3a_branch2a')
(40, 'activation_11')
(41, 'res3a_branch2b')
(42, 'bn3a_branch2b')
(43, 'activation_12')
(44, 'res3a_branch2c')
(45, 'res3a_branch1')
(46, 'bn3a_branch2c')
(47, 'bn3a_branch1')
(48, 'merge_4')
(49, 'activation_13')
(50, 'res3b_branch2a')
(51, 'bn3b_branch2a')
(52, 'activation_14')
(53, 'res3b_branch2b')
(54, 'bn3b_branch2b')
(55, 'activation_15')
(56, 'res3b_branch2c')
(57, 'bn3b_branch2c')
(58, 'merge_5')
(59, 'activation_16')
(60, 'res3c_branch2a')
(61, 'bn3c_branch2a')
(62, 'activation_17')
(63, 'res3c_branch2b')
(64, 'bn3c_branch2b')
(65, 'activation_18')
(66, 'res3c_branch2c')
(67, 'bn3c_branch2c')
(68, 'merge_6')
(69, 'activation_19')
(70, 'res3d_branch2a')
(71, 'bn3d_branch2a')
(72, 'activation_20')
(73, 'res3d_branch2b')
(74, 'bn3d_branch2b')
(75, 'activation_21')
(76, 'res3d_branch2c')
(77, 'bn3d_branch2c')
(78, 'merge_7')
(79, 'activation_22')
(80, 'res4a_branch2a')
(81, 'bn4a_branch2a')
(82, 'activation_23')
(83, 'res4a_branch2b')
(84, 'bn4a_branch2b')
(85, 'activation_24')
(86, 'res4a_branch2c')
(87, 'res4a_branch1')
(88, 'bn4a_branch2c')
(89, 'bn4a_branch1')
(90, 'merge_8')
(91, 'activation_25')
(92, 'res4b_branch2a')
(93, 'bn4b_branch2a')
(94, 'activation_26')
(95, 'res4b_branch2b')
(96, 'bn4b_branch2b')
(97, 'activation_27')
(98, 'res4b_branch2c')
(99, 'bn4b_branch2c')
(100, 'merge_9')
(101, 'activation_28')
(102, 'res4c_branch2a')
(103, 'bn4c_branch2a')
(104, 'activation_29')
(105, 'res4c_branch2b')
(106, 'bn4c_branch2b')
(107, 'activation_30')
(108, 'res4c_branch2c')
(109, 'bn4c_branch2c')
(110, 'merge_10')
(111, 'activation_31')
(112, 'res4d_branch2a')
(113, 'bn4d_branch2a')
(114, 'activation_32')
(115, 'res4d_branch2b')
(116, 'bn4d_branch2b')
(117, 'activation_33')
(118, 'res4d_branch2c')
(119, 'bn4d_branch2c')
(120, 'merge_11')
(121, 'activation_34')
(122, 'res4e_branch2a')
(123, 'bn4e_branch2a')
(124, 'activation_35')
(125, 'res4e_branch2b')
(126, 'bn4e_branch2b')
(127, 'activation_36')
(128, 'res4e_branch2c')
(129, 'bn4e_branch2c')
(130, 'merge_12')
(131, 'activation_37')
(132, 'res4f_branch2a')
(133, 'bn4f_branch2a')
(134, 'activation_38')
(135, 'res4f_branch2b')
(136, 'bn4f_branch2b')
(137, 'activation_39')
(138, 'res4f_branch2c')
(139, 'bn4f_branch2c')
(140, 'merge_13')
(141, 'activation_40')
(142, 'res5a_branch2a')
(143, 'bn5a_branch2a')
(144, 'activation_41')
(145, 'res5a_branch2b')
(146, 'bn5a_branch2b')
(147, 'activation_42')
(148, 'res5a_branch2c')
(149, 'res5a_branch1')
(150, 'bn5a_branch2c')
(151, 'bn5a_branch1')
(152, 'merge_14')
(153, 'activation_43')
(154, 'res5b_branch2a')
(155, 'bn5b_branch2a')
(156, 'activation_44')
(157, 'res5b_branch2b')
(158, 'bn5b_branch2b')
(159, 'activation_45')
(160, 'res5b_branch2c')
(161, 'bn5b_branch2c')
(162, 'merge_15')
(163, 'activation_46')
(164, 'res5c_branch2a')
(165, 'bn5c_branch2a')
(166, 'activation_47')
(167, 'res5c_branch2b')
(168, 'bn5c_branch2b')
(169, 'activation_48')
(170, 'res5c_branch2c')
(171, 'bn5c_branch2c')
(172, 'merge_16')
(173, 'activation_49')
(174, 'avg_pool')
(175, 'flatten_1')
(176, 'fc1000')

'''

for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

img_path = '/imatge/mcompri/Desktop/THESIS/Places-CNN/ESEMPIO_coffe2keras/keras-master/keras/caffe/modello_prova/training/images_training/'

# add a global spatial average pooling layer
x = base_model.output

# let's add a fully-connected layer
fc1 = Dense(1000,activation='tanh')(x)
b2=BatchNormalization(axis=1)(fc1)
d1=Dropout(0.6)(b2)
fc2 = Dense(1000, activation='tanh', name = 'fc2')(d1)
b3=BatchNormalization(axis=1)(fc2)
d2 =Dropout(0.6)(b3)
predictions = Dense(17, activation='sigmoid')(d2)
# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional ResNet50 layers
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
				#x = image.img_to_array(img)
				#print(train_datagen.shape)  #1,3,224,224
				im_array.append(img)
				#train_datagen_new = train_datagen.reshape(train_datagen, (1,4)) # output: error
				#print(train_datagen.shape)

im_array = np.array(im_array)
trainFeatures = im_array


# LABELS LOAD

dataframe = pd.read_csv("multi_label_training_1680.csv", header=None)
dataset = dataframe.values
r = np.array(dataset)


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
for layer in model.layers[:162]:
   layer.trainable = False
for layer in model.layers[162:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0000001, momentum=0.9), loss='binary_crossentropy')
# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
#model.fit_generator(...)
model.fit(trainFeatures,r,batch_size=32,validation_split=0.33,nb_epoch=10,verbose=2)

print("saving model and weights")
model_json = model.to_json()

with open("ResNet50.json", "w") as json_file:
	json_file.write(model_json)
print("saving...")
model.save_weights('resnet50_weights.h5')
