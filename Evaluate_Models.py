######################################## pwd
import sys
pwd=sys.path[0]
########################################

######################################## Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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

########## load set
### get user input
print('Choose tested set')
file_name_path = filedialog.askopenfilename()
file_name_set=Main_Func.Get_Name_From_File_Path(file_name_path)

### load set
filehandler  = open(file_name_path,'rb')
set = pickle.load(filehandler)
filehandler.close()

### unpack set
features, labels=Main_Func.Unpack_Set(set)

########## load models
### load NN
file_name=pwd+'\\'+ 'Model_NN' +'.model'
model_NN = tf.keras.models.load_model(file_name)

### load CNN
file_name=pwd+'\\'+ 'Model_CNN' +'.model'
model_CNN = tf.keras.models.load_model(file_name)

### load SVM 1
file_name=pwd+'\\'+ 'Model_SVM_1' +'.obj'
filehandler  = open(file_name,'rb')
model_SVM_1 = pickle.load(filehandler)
filehandler.close()

### load SVM 2
file_name=pwd+'\\'+ 'Model_SVM_2' +'.obj'
filehandler  = open(file_name,'rb')
model_SVM_2 = pickle.load(filehandler)
filehandler.close()

########## evaluate
### evaluate NN
print('--------------------')
print('--------------------')
print('--------------------')
print('Test set is:', file_name_set)
print('--------------------')
print('NN:')
_, succes_rate_NN = model_NN.evaluate(features, labels)
succes_rate_NN*=100
print('Success Rate=', ' %0.2f ' % succes_rate_NN ,'%')

### evaluate CNN
features_cnn = features.reshape(features.shape[0], 28, 28, 1) # features reshape for cnn
#
print('--------------------')
print('CNN:')
_, succes_rate_CNN = model_CNN.evaluate(features_cnn, labels)
succes_rate_CNN*=100
print('Success Rate=', ' %0.2f ' % succes_rate_CNN ,'%')

### flat set
features_flat = features.reshape(features.shape[0], features[0].size)

### evaluate SVM 1
succes_rate_svm_1=model_SVM_1.score(features_flat, labels)
succes_rate_svm_1*=100
print('--------------------')
print('SVM 1:')
print('Success Rate=', ' %0.2f ' % succes_rate_svm_1 ,'%')

### evaluate SVM 2
succes_rate_svm_2=model_SVM_2.score(features_flat, labels)
succes_rate_svm_2*=100
print('--------------------')
print('SVM 2:') 
print('Success Rate=', ' %0.2f ' % succes_rate_svm_2 ,'%')

### plot
x_axis = [0, 1, 2, 3]
y_axis = [succes_rate_NN, succes_rate_CNN, succes_rate_svm_1, succes_rate_svm_2]
plot_label=['NN', 'CNN', 'SVM1-'+model_SVM_1.kernel, 'SVM2-'+model_SVM_2.kernel]
plot_color=['red', 'green', 'blue', 'yellow']
plot_arr=Main_Func.Init_List(4)
#
for i in range(len(x_axis)):
    plot_arr[i]=plt.stem([x_axis[i]], [y_axis[i]], label=plot_label[i])
    plt.setp(plot_arr[i], markersize=10, color=plot_color[i], markeredgewidth=2)
    plt.text(x_axis[i]+0.05, y_axis[i], str(' %0.2f ' % y_axis[i])+'%', fontsize=12)
plt.title('Tested Set is: '+file_name_set)
plt.legend()
plt.show()

### hold cmd
print('--------------------')
input('Press Enter to exit...')