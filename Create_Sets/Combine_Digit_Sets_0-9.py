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

########## load files
num_of_digits=10
file_name=[]
set_list=Main_Func.Init_List(num_of_digits)
for i in range(num_of_digits):
    file_name.append(path_Digits_Data+'\\'+ str(i) + '.obj')
    filehandler  = open(file_name[i],'rb')
    set_list[i] = pickle.load(filehandler)
    filehandler.close()

########## concatenate
features, labels=Main_Func.Unpack_Set(set_list[0])
for i in range(1,num_of_digits):
    features_cur, labels_cur=Main_Func.Unpack_Set(set_list[i])
    features = np.concatenate((features,features_cur), axis = 0) 
    labels = np.concatenate((labels,labels_cur), axis = 0) 

########## pack
digits_set=Main_Func.Pack_Set(features, labels)

########## save to file
file_name=input('Enter output file name: ')
file_name_path=path_Sets+'\\'+ file_name +'.obj'
filehandler = open(file_name_path,"wb")
pickle.dump(digits_set,filehandler)
filehandler.close()
#
print( 'Files 0-9 were concatenated and saved to file:', file_name)
