######################################## Imports
import numpy as np
from scipy import misc
from skimage import color
import tensorflow as tf
########################################

####################
def Picture_Proccess(img, pixel_norm):
    img = misc.imresize(img, (pixel_norm,pixel_norm)) # 3 channel RGB 28x28
    gray = color.rgb2gray(img) # 1 channel grayscale
    gray_scaled = misc.bytescale(gray, high=255, low=0) # scaled to 255
    pic = tf.keras.utils.normalize(gray_scaled, axis =1) # norm
    return pic
####################

####################
def Pack_Set(features, labels):  
    list=Init_List(2)
    list[0]=features
    list[1]=labels
    return list # list[0]= features, list[1]= labels
####################

####################
def Unpack_Set(list):
    return list[0], list[1] # return: features, labels
####################

####################    
def Count_Digits(labels):
    list_digits=Init_List(10)
    num_of_labels=np.size(labels)
    for i in range(num_of_labels):
        cur_label=labels[i]
        list_digits[cur_label]+=1
    return list_digits
####################

####################
def Init_List(n):
    list=[]
    for i in range(0,n):
        list.append(0)
    return list
####################

####################
def Get_Name_From_File_Path(file_path):
    str_path=file_path.split("/")
    file_name=str_path[len(str_path)-1]
    str_path=file_name.split(".")
    file_name=str_path[0]
    return file_name
####################

################################################################################ Unused functions

####################
def Flat_For_Sample(pic):
    length, width=np.shape(pic)
    flat_pic=np.zeros(length*width)
    index=0
    for i in range(length):
        for j in range(width):
            flat_pic[index]=pic[i][j]
            index+=1

    return flat_pic

####################

####################
def Flat_Set(features):
    num_of_samples, length, width=np.shape(features)
    features_flat=np.zeros((num_of_samples, length*width))
    for i in range(num_of_samples):
        features_flat[i]=Flat_For_Sample(features[i])
    return features_flat     

####################

####################
def Evaluate_SVM(model, features_flat, labels):
    count_success=0
    num_of_features,_=np.shape(features_flat)
    for i in range(num_of_features):
        prediction=model.predict([features_flat[i]])
        prediction=prediction[0]
        if (prediction == labels[i]):
            count_success+=1
    succes_rate = (count_success/num_of_features)*100
    return succes_rate
####################