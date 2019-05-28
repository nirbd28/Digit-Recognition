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
file_name=Main_Func.Get_Name_From_File_Path(file_name_path)
print('Trained set is:', file_name)
print('-----')

##### load sets
filehandler  = open(file_name_path,'rb')
set = pickle.load(filehandler)
filehandler.close()

##### unpack
features, labels=Main_Func.Unpack_Set(set)

########## create model
model = tf.keras.models.Sequential() 
###
model.add(tf.keras.layers.Flatten()) # flatten the input
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu)) # first hidden layer with 128 neroun and relu activion function
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu)) # secend hidden layer with 128 neroun and relu activion function
model.add(tf.keras.layers.Dense(10,activation= tf.nn.softmax)) # output layer with 10 neron  becuse we have 10 digits (0-9)
###
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']) 
model.fit(features, labels, epochs=5) # to train the model how match epochs we want
###
print('Model Trained')

##### final evaluate
print('--------------------')
print('Evaluation')
cost, succes_rate = model.evaluate(features, labels)
succes_rate*=100
print('Cost=',' %0.1f ' % cost,',', 'Success Rate=', ' %0.1f ' % succes_rate ,'%')

##### save
file_name='Model_NN'
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

