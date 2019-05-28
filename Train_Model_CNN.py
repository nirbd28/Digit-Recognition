######################################## pwd
import sys
pwd=sys.path[0]
########################################

######################################## Imports
import numpy as np
import tensorflow as tf
import pickle
###
import tkinter as tk
from tkinter import filedialog
#
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
########################################

######################################## Import from folder
folder_name='Functions'
path_name_new=pwd + '\\' + folder_name
sys.path.append(path_name_new)
##### Import Files
import Main_Func
########################################

################################################################################
################################################################################
################################################################################

##### get user input
print('Choose train set')
file_name_path = filedialog.askopenfilename()
#
file_name=Main_Func.Get_Name_From_File_Path(file_name_path)
print('Trained set is:', file_name)
print('-----')

##### load sets
filehandler  = open(file_name_path,'rb')
set = pickle.load(filehandler)
filehandler.close()

##### unpack
features, labels=Main_Func.Unpack_Set(set)

##### features reshape for cnn
features_cnn = features.reshape(features.shape[0], 28, 28, 1)

########## create model
input_layer=KL.Input(shape=(28,28,1))
###
c=KL.Conv2D(32, (3,3), padding='valid',activation=tf.nn.relu)(input_layer)
m=KL.MaxPool2D( (2, 2) , (2, 2)  )(c)
d=KL.Dropout(0.5)(m)
c=KL.Conv2D(64, (3,3), padding='valid',activation=tf.nn.relu)(d)
m=KL.MaxPool2D( (2, 2) , (2, 2)  )(c)
d=KL.Dropout(0.5)(m)
###
f=KL.Flatten()(d)
output_layer= KL.Dense(10, activation=tf.nn.softmax)(f)
###
model= KM.Model(input_layer, output_layer)
###
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam", metrics=['accuracy'])
model.fit(features_cnn, labels, epochs=5)
###
print('Model Trained')

##### final evaluate
print('--------------------')
print('Evaluation')
cost, succes_rate = model.evaluate(features_cnn, labels)
succes_rate*=100
print('Cost=',' %0.1f ' % cost,',', 'Success Rate=', ' %0.1f ' % succes_rate ,'%')

##### save
file_name='Model_CNN'
file_name_path=pwd+'\\'+ file_name +'.model'
model.save(file_name_path)
###
print('Model was saved to file:', file_name)

### model summary
print('--------------------')
print('Model Summary:')
model.summary()

### hold cmd
print('--------------------')
input('Press Enter to exit...')