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
import pickle
###
import tkinter as tk
from tkinter import filedialog
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

##### get user input
print('Choose first set')
file_path_1 = filedialog.askopenfilename()
#
print('Choose second set')
file_path_2 = filedialog.askopenfilename()

##### load sets
filehandler  = open(file_path_1,'rb')
set_1 = pickle.load(filehandler)
filehandler.close()
###
filehandler  = open(file_path_2,'rb')
set_2 = pickle.load(filehandler)
filehandler.close()

##### concatenate
set_1_features, set_1_labels=Main_Func.Unpack_Set(set_1)
set_2_features, set_2_labels=Main_Func.Unpack_Set(set_2)
###
features = np.concatenate((set_1_features, set_2_features), axis = 0)
labels = np.concatenate((set_1_labels, set_2_labels), axis = 0)

##### continue?
while 1:
    print('Choose set, click cancel to exit')
    file_path = filedialog.askopenfilename()
    if (file_path==''):
        break
    filehandler  = open(file_path_1,'rb')
    set_i = pickle.load(filehandler)
    filehandler.close()
    #
    set_features, set_labels=Main_Func.Unpack_Set(set_i)
    #
    features = np.concatenate((features, set_features), axis = 0)
    labels = np.concatenate((labels, set_labels), axis = 0)

##### pack set
set=Main_Func.Pack_Set(features, labels)

##### save
file_name=input('Enter output file name: ')
file_name_path=path_Sets+'\\'+ file_name +'.obj'
filehandler = open(file_name_path,"wb")
pickle.dump(set,filehandler)
filehandler.close()
#
print('Saved to file:', file_name)

