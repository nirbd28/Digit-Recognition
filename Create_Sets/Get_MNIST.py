######################################## pwd
import sys
pwd=sys.path[0]
########################################

######################################## Main project path
str_path=pwd.split("\\")
path_main=str_path[0]
for i in range(1,len(str_path)-1):
    path_main=path_main+'\\'+str_path[i]
########################################

######################################## other paths
path_Digits_Data=pwd +'\\'+'Digits_Data'
path_Sets=pwd +'\\'+'Sets'
########################################

######################################## Imports
import numpy as np
import tensorflow as tf
import pickle
########################################

######################################## Import from folder
folder_name='Functions'
path_name_new=path_main + '\\' + folder_name
sys.path.append(path_name_new)
##### Import Files
import Main_Func
########################################

################################################################################
################################################################################
################################################################################

########## get dataset
MNIST =tf.keras.datasets.mnist 
(train_features , train_labels) , (test_features , test_labels) = MNIST.load_data()

########## normalize
train_features = tf.keras.utils.normalize(train_features, axis =1)
test_features = tf.keras.utils.normalize(test_features, axis =1)

########## pack sets
mnist_train=Main_Func.Pack_Set(train_features, train_labels)
mnist_test=Main_Func.Pack_Set(test_features, test_labels)

########## save file
file_name_train='mnist_train'
file_name_test='mnist_test'
file_name=[path_Sets+'\\'+file_name_train+'.obj',path_Sets+'\\'+file_name_test+'.obj']
file_list = [mnist_train, mnist_test]
###
for i in range (len(file_name)): 
    filehandler = open(file_name[i],"wb")
    pickle.dump(file_list[i],filehandler)
    filehandler.close()
    
##########
print('MNIST train set was saved to file:', file_name_train)
print('MNIST test set was saved to file:', file_name_test)

