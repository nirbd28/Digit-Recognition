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
###
import random
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

################################################################################ functions

##########
def Concatenate_Image(features, pic):
    pic = np.expand_dims(pic, axis=0)
    features = np.concatenate((features,pic), axis = 0) 
    return features
##########

##########
def Split_Set_By_List(set, list):
    ### define
    features_new=[]
    labels_new=np.zeros(len(list))
    labels_new = labels_new.astype(np.uint8)

    ### unpack
    features, labels=Main_Func.Unpack_Set(set)

    ### concatenate
    pic = np.expand_dims(features[list[0]], axis=0)
    features_new = pic 
    labels_new[0]= labels[list[0]]

    for i in range(1,len(list)):
        cur_feature=features[list[i]]
        cur_label=labels[list[i]]
        ###
        features_new=Concatenate_Image(features_new, cur_feature)
        labels_new[i]=cur_label
    
    ### pack
    set=Main_Func.Pack_Set(features_new, labels_new)

    return set
##########

##########
def Randomize_Set_Index(labels, list_labels_idx, percent):
    ### init 
    list_of_train=Init_List_Ascending_Digits(len(labels))
    list_of_test=[]

    ### randomize per digit
    for digit_i in range(len(list_labels_idx)):
        share_of_test_set=round(len(list_labels_idx[digit_i])*(percent/100))
        for i in range(share_of_test_set):
            rnd=random.choice(list_labels_idx[digit_i])
            list_labels_idx[digit_i].remove(rnd)
            list_of_train.remove(rnd)
            list_of_test.append(rnd)

    return list_of_train, list_of_test
##########

##########
def Get_Digits_Index_Labels(labels):
    list_labels_idx=Main_Func.Init_List(10)
    ### init
    for i in range(len(list_labels_idx)):
        list_labels_idx[i]=[]
    ### get digits index labels
    for label_i in range(len(labels)):
        list_labels_idx[labels[label_i]].append(label_i)
    return list_labels_idx        
##########

##########
def Init_List_Ascending_Digits(n):
    list=[]
    for i in range(n):
        list.append(i)
    return list
##########

################################################################################ main

##### get user input
print('Choose set to split')
file_path = filedialog.askopenfilename()
file_name_set=Main_Func.Get_Name_From_File_Path(file_path)

##### load sets
filehandler  = open(file_path,'rb')
set = pickle.load(filehandler)
filehandler.close()

### unpack
features, labels=Main_Func.Unpack_Set(set)

### choose precent of split (how much to give to test set per word)
percent=int(input('Enter split percentage(%):'))

############### split

### put index of digit
list_labels_idx=Get_Digits_Index_Labels(labels)

### randomize
list_of_train, list_of_test=Randomize_Set_Index(labels, list_labels_idx, percent)

############### get set from lists

### get sets
set_train=Split_Set_By_List(set, list_of_train)
set_test=Split_Set_By_List(set, list_of_test)

### save sets
set_train_name=file_name_set+'_train'
set_test_name=file_name_set+'_test'

# save train set
file_name_path=path_Sets+'\\'+ set_train_name +'.obj'
filehandler = open(file_name_path,"wb")
pickle.dump(set_train,filehandler)
filehandler.close()

# save test set
file_name_path=path_Sets+'\\'+ set_test_name +'.obj'
filehandler = open(file_name_path,"wb")
pickle.dump(set_test,filehandler)
filehandler.close()

### print
print(file_name_set, 'was split into', set_train_name, 'and', set_test_name)
print('Split percentage is:', str(percent)+'%','(toward test set)' )