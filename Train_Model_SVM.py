######################################## pwd
import sys
pwd=sys.path[0]
########################################

######################################## Imports
import numpy as np
import pickle
###
import tkinter as tk
from tkinter import filedialog
###
from sklearn.svm import SVC
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

##### flat set
features_flat = features.reshape(features.shape[0], features[0].size)

##### choose kernels
kernel_options=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
#
model_1_kernel='rbf'
model_2_kernel='linear'

########## create model 1
print('Training model 1 \n . \n . \n .')
model_1 = SVC(C=5, gamma = 0.05, kernel=model_1_kernel)# RBF
model_1.fit(features_flat, labels)
print('Model 1 Trained')

########## create model 2
print('Training model 2 \n . \n . \n .')
model_2 = SVC(C=5, kernel=model_2_kernel)# linear
model_2.fit(features_flat, labels)
print('Model 2 Trained')

##### save model 1
file_name_svm_1='Model_SVM_1'
file_name_svm_1_path=pwd+'\\'+ file_name_svm_1 +'.obj'
filehandler = open(file_name_svm_1_path,"wb")
pickle.dump(model_1,filehandler)
filehandler.close()

##### save model 2
file_name_svm_2='Model_SVM_2'
file_name_svm_2_path=pwd+'\\'+ file_name_svm_2 +'.obj'
filehandler = open(file_name_svm_2_path,"wb")
pickle.dump(model_2,filehandler)
filehandler.close()
###
print('SVM model 1 saved to file:', file_name_svm_1)
print('SVM model 2 saved to file:', file_name_svm_2)

### hold cmd
print('--------------------')
input('Press Enter to exit...')