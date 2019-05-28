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
from tkinter import *
from tkinter import filedialog
###
from sklearn.svm import SVC
###
import cv2
########################################

######################################## Import from folder
folder_name='Functions'
path_name_new=pwd + '\\' + folder_name
sys.path.append(path_name_new)
##### Import Files
import Main_Func, Define
########################################

################################################################################
################################################################################
################################################################################

################################################################################ load models

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

########## Instructions
print('--------------------')
print('Instructions:')
print('Right Click - clear')
print('Double Left Click - exit program')

################################################################################ functions

########## inputs 
pixel_window=Define.Parameters.pixel_window
pixel_norm=Define.Parameters.pixel_norm
draw_radius=Define.Parameters.draw_radius
#
model_str_arr=['NN', 'CNN', 'SVM_1', 'SVM_2']

### parameters
img=[]
features=[]
counter=0 
flag_exit=0
###
drawing = False 

########## draw circle
def draw_circle(event,x,y,flags,param):
    global drawing, img, features
    global counter, flag_exit

    if event == cv2.EVENT_LBUTTONDBLCLK: # double click for exit and save data 
        flag_exit=1

    if event == cv2.EVENT_RBUTTONDOWN: # click for clear img
        ### new window
        img=np.full((pixel_window,pixel_window,3),0,np.float64)
      
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),draw_radius,(255,255,255),-1)
                
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    
############################################################

########## Print delay time
show_time=40 
show_time_i=0

########## main draw
def Main_Draw(model_str):
    global show_time_i, show_time 
    global flag_exit
    global img
    
    ### new window
    img=np.full((pixel_window,pixel_window,3),0,np.float64) # new window

    ### draw loop
    while(flag_exit==0): 
        ### show window
        cv2.imshow(model_str,img)
        cv2.namedWindow(model_str)
        cv2.setMouseCallback(model_str,draw_circle) # call draw function

        if show_time_i==show_time:
            ### picture proccesing
            pic=Main_Func.Picture_Proccess(img, pixel_norm)

            ### predict
            prediction=Prediction_Func(pic, model_str)
            ###
            show_time_i=0
            print("---")
            print(prediction)

        ### update show counter   
        show_time_i+=1

        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

    print("----------------------------------------")
    flag_exit=0
##########

########## predict function
def Prediction_Func(pic, model_str):
    ### NN
    if (model_str==model_str_arr[0]):
        model=model_NN
        pic=np.array([pic]) # convert to np array
        prediction = model.predict(pic)
        prediction=np.argmax(prediction)

    ### CNN    
    elif (model_str==model_str_arr[1]):
        model=model_CNN
        pic = np.expand_dims(pic, axis=2)
        pic = np.expand_dims(pic, axis=0)
        prediction = model.predict(pic)
        prediction = np.argmax(prediction)

    ### SVM 1    
    elif (model_str==model_str_arr[2]):
        model=model_SVM_1
        flat_pic = pic.reshape(pic.size, )# flatten
        prediction=model.predict([flat_pic])
        prediction=prediction[0]

    ### SVM 2    
    elif (model_str==model_str_arr[3]):
        model=model_SVM_2
        flat_pic = pic.reshape(pic.size, )# flatten
        prediction=model.predict([flat_pic])
        prediction=prediction[0]

    ###
    return prediction

##########

##### sub draw
def Test_NN():
    model_str=model_str_arr[0]
    Main_Draw(model_str)
#####
def Test_CNN():
    model_str=model_str_arr[1]
    Main_Draw(model_str)
#####
def Test_SVM_1():
    model_str=model_str_arr[2]
    Main_Draw(model_str)
#####
def Test_SVM_2():
    model_str=model_str_arr[3]
    Main_Draw(model_str)
#####

################################################################################ gui

### main
root = Tk()
root.title('GUI')
root.geometry("350x200")
root.configure(bg='white')

### frames 
frame1 = Frame(root)
frame1.pack()
frame2 = Frame(root)
frame2.pack()
frame3 = Frame(root)
frame3.pack()
frame4 = Frame(root)
frame4.pack()
frame5 = Frame(root)
frame5.pack()
frame6 = Frame(root)
frame6.pack()
frame7 = Frame(root)
frame7.pack()
frame8 = Frame(root)
frame8.pack()

### frame1
label_title = Label(frame1, text="Digits Recognition", fg="white", bg="black", width=20, height=3 )
label_title.config(font=("Courier", 20))
label_title.pack( side = LEFT)

### frame2
btn_NN = Button(frame2, text="NN", fg="black", bg="red", width=10, height=3, command=Test_NN)
btn_NN.pack(side = LEFT)
#
btn_CNN = Button(frame2, text="CNN", fg="black", bg="green", width=10, height=3, command=Test_CNN)
btn_CNN.pack(side = LEFT)
#
btn_SVM_1 = Button(frame2, text="SVM 1", fg="black", bg="blue", width=10, height=3, command=Test_SVM_1)
btn_SVM_1.pack(side = LEFT)
#
btn_SVM_2 = Button(frame2, text="SVM 2", fg="black", bg="yellow", width=10, height=3, command=Test_SVM_2)
btn_SVM_2.pack(side = LEFT)

########## main loop
root.mainloop()
