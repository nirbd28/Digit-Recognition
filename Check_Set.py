######################################## pwd
import sys
pwd=sys.path[0]
########################################

######################################## other paths
path_Digits_Data=pwd +'\\'+'Digits_Data'
########################################

######################################## Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
###
from sklearn.svm import SVC
###
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import *
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

##### global
global set ,features, labels
global model_NN, model_SVM_1, model_SVM_2

#################### load models
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

################################################################################ functions

####################
def Load_Set():
    global set, features, labels
    ### enable show pic btn
    btn_show_pic.config(state=ACTIVE)

    ### get user input
    print('Choose set')
    file_path = filedialog.askopenfilename()
    file_name=Main_Func.Get_Name_From_File_Path(file_path)

    ### set label txt
    label_set_name_val.configure(text=file_name)

    ### load set
    filehandler  = open(file_path,'rb')
    set = pickle.load(filehandler)
    filehandler.close()

    ### unpack
    features, labels=Main_Func.Unpack_Set(set)

    ### digit counter
    list_digits=Main_Func.Count_Digits(labels)

    ### set label txt
    label_digit_name_val=[label_0_name_val, label_1_name_val, label_2_name_val, label_3_name_val, label_4_name_val, label_5_name_val, label_6_name_val, label_7_name_val, label_8_name_val, label_9_name_val]
    for i in range(len(label_digit_name_val)):
        label_digit_name_val[i].configure(text=list_digits[i])

    ### set label txt - total samples
    label_digit_total_val.configure(text=len(labels))

####################

####################
def Show_Pic():
    global set ,features, labels

    ### get value of entry
    idx = int(entry_pic_index.get())
    if ((idx>len(labels)-1) or (idx<0)):
        messagebox.showinfo("Error", "Choose index whitin range of 0 to "+str(len(labels)-1))
    else:    
        pic=features[idx]
        label=labels[idx]

        ### predict outputs
        prediction_NN, prediction_CNN, prediction_svm_1, prediction_svm_2 = Predict_Models_Output(pic)
        print('--------------------')
        print('--------------------')
        print('--------------------')
        print('NN Output=', prediction_NN)
        print('CNN Output=', prediction_CNN)
        print('SVM 1 Output=', prediction_svm_1)
        print('SVM 2 Output=', prediction_svm_2)

        ### plot
        plt.imshow(pic, cmap = plt.cm.binary) 
        plt.title('Label='+str(label))
        plt.show()
        
####################

####################
def Predict_Models_Output(pic):
    global model_NN, model_NN, model_SVM_1, model_SVM_2
    
    ### for NN
    pic_NN=np.array([pic]) # convert to np array
    prediction_NN = model_NN.predict(pic_NN)
    prediction_NN=np.argmax(prediction_NN)

    ### for CNN
    pic = np.expand_dims(pic, axis=2)
    pic = np.expand_dims(pic, axis=0)
    prediction_CNN = model_CNN.predict(pic)
    prediction_CNN = np.argmax(prediction_CNN)

    ### for SVM 1
    flat_pic = pic.reshape(pic.size, )# flatten
    prediction_svm_1=model_SVM_1.predict([flat_pic])
    prediction_svm_1=prediction_svm_1[0]

    ### for SVM 2
    flat_pic = pic.reshape(pic.size, )# flatten
    prediction_svm_2=model_SVM_2.predict([flat_pic])
    prediction_svm_2=prediction_svm_2[0]

    return prediction_NN, prediction_CNN, prediction_svm_1, prediction_svm_2

####################

####################
def Entry_Up():
    idx = int(entry_pic_index.get())
    idx+=1
    entry_pic_index.delete(0,END)
    entry_pic_index.insert(0,idx)

def Entry_Down():
    idx = int(entry_pic_index.get())
    idx-=1
    entry_pic_index.delete(0,END)
    entry_pic_index.insert(0,idx)


####################

################################################################################ gui

### main
root = Tk()
root.title('Check Set')
root.geometry("450x600")
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
frame9 = Frame(root)
frame9.pack()
frame10 = Frame(root)
frame10.pack()
frame11 = Frame(root)
frame11.pack()
frame12 = Frame(root)
frame12.pack()
frame13 = Frame(root)
frame13.pack()

### frame1
label_title = Label(frame1, text="Check Set", fg="white", bg="purple", width=20, height=3 )
label_title.config(font=("Courier", 20))
label_title.pack( side = LEFT)

### frame2
btn_load_set = Button(frame2, text="Load Set", fg="black", bg="green", width=10, height=3, command=Load_Set)
btn_load_set.pack(side = LEFT)

### frame3
label_set_name = Label(frame3, text="Set Name:", fg="black", bg="white", width=10, height=3 )
label_set_name.config(font=("Courier", 10))
label_set_name.pack( side = LEFT)
#
label_set_name_val = Label(frame3, fg="red", bg="white", width=20, height=3 )
label_set_name_val.config(font=("Courier", 10))
label_set_name_val.pack( side = LEFT)

### frame4
label_0_name = Label(frame4, text="Digit 0", fg="black", bg="white", width=10, height=3 )
label_0_name.config(font=("Courier", 10))
label_0_name.pack( side = LEFT)
#
label_1_name = Label(frame4, text="Digit 1", fg="black", bg="white", width=10, height=3 )
label_1_name.config(font=("Courier", 10))
label_1_name.pack( side = LEFT)
#
label_2_name = Label(frame4, text="Digit 2", fg="black", bg="white", width=10, height=3 )
label_2_name.config(font=("Courier", 10))
label_2_name.pack( side = LEFT)
#
label_3_name = Label(frame4, text="Digit 3", fg="black", bg="white", width=10, height=3 )
label_3_name.config(font=("Courier", 10))
label_3_name.pack( side = LEFT)
#
label_4_name = Label(frame4, text="Digit 4", fg="black", bg="white", width=10, height=3 )
label_4_name.config(font=("Courier", 10))
label_4_name.pack( side = LEFT)
#

### frame5
back_ground='blue'
label_0_name_val = Label(frame5, text="", fg="white", bg=back_ground, width=10, height=3 )
label_0_name_val.config(font=("Courier", 10))
label_0_name_val.pack( side = LEFT)
#
label_1_name_val = Label(frame5, text="", fg="white", bg=back_ground, width=10, height=3 )
label_1_name_val.config(font=("Courier", 10))
label_1_name_val.pack( side = LEFT)
#
label_2_name_val = Label(frame5, text="", fg="white", bg=back_ground, width=10, height=3 )
label_2_name_val.config(font=("Courier", 10))
label_2_name_val.pack( side = LEFT)
#
label_3_name_val = Label(frame5, text="", fg="white", bg=back_ground, width=10, height=3 )
label_3_name_val.config(font=("Courier", 10))
label_3_name_val.pack( side = LEFT)
#
label_4_name_val = Label(frame5, text="", fg="white", bg=back_ground, width=10, height=3 )
label_4_name_val.config(font=("Courier", 10))
label_4_name_val.pack( side = LEFT)
#

### frame6
frame6.config(bg='red', width=450, height=5)

### frame7
label_5_name = Label(frame7, text="Digit 5", fg="black", bg="white", width=10, height=3 )
label_5_name.config(font=("Courier", 10))
label_5_name.pack( side = LEFT)
#
label_6_name = Label(frame7, text="Digit 6", fg="black", bg="white", width=10, height=3 )
label_6_name.config(font=("Courier", 10))
label_6_name.pack( side = LEFT)
#
label_7_name = Label(frame7, text="Digit 7", fg="black", bg="white", width=10, height=3 )
label_7_name.config(font=("Courier", 10))
label_7_name.pack( side = LEFT)
#
label_8_name = Label(frame7, text="Digit 8", fg="black", bg="white", width=10, height=3 )
label_8_name.config(font=("Courier", 10))
label_8_name.pack( side = LEFT)
#
label_9_name = Label(frame7, text="Digit 9", fg="black", bg="white", width=10, height=3 )
label_9_name.config(font=("Courier", 10))
label_9_name.pack( side = LEFT)
#

### frame8
back_ground='blue'
label_5_name_val = Label(frame8, text="", fg="white", bg=back_ground, width=10, height=3 )
label_5_name_val.config(font=("Courier", 10))
label_5_name_val.pack( side = LEFT)
#
label_6_name_val = Label(frame8, text="", fg="white", bg=back_ground, width=10, height=3 )
label_6_name_val.config(font=("Courier", 10))
label_6_name_val.pack( side = LEFT)
#
label_7_name_val = Label(frame8, text="", fg="white", bg=back_ground, width=10, height=3 )
label_7_name_val.config(font=("Courier", 10))
label_7_name_val.pack( side = LEFT)
#
label_8_name_val = Label(frame8, text="", fg="white", bg=back_ground, width=10, height=3 )
label_8_name_val.config(font=("Courier", 10))
label_8_name_val.pack( side = LEFT)
#
label_9_name_val = Label(frame8, text="", fg="white", bg=back_ground, width=10, height=3 )
label_9_name_val.config(font=("Courier", 10))
label_9_name_val.pack( side = LEFT)
#

### frame10
label_digit_total = Label(frame10, text="Total Samples:", fg="black", bg="white", width=15, height=3 )
label_digit_total.config(font=("Courier", 10))
label_digit_total.pack( side = LEFT)
#
label_digit_total_val = Label(frame10, fg="red", bg="white", width=10, height=3 )
label_digit_total_val.config(font=("Courier", 10))
label_digit_total_val.pack( side = LEFT)

### frame11
frame11.config(bg='red', width=450, height=5)

### frame12
label_pic_index = Label(frame12, text="Enter Pic Index:", fg="black", bg="white", width=15, height=3 )
label_pic_index.config(font=("Courier", 10))
label_pic_index.pack( side = LEFT)
#
btn_down = Button(frame12, text="↓", fg="black", bg="red", width=2, height=1, command=Entry_Down)
btn_down.pack(side = LEFT)
#
entry_pic_index = Entry(frame12, bg="white", width =6)
entry_pic_index.insert(0, "0")
entry_pic_index.pack(side = LEFT)
#
btn_up = Button(frame12, text="↑", fg="black", bg="red", width=2, height=1, command=Entry_Up)
btn_up.pack(side = LEFT)
#
btn_show_pic = Button(frame12, text="Show Pic", fg="black", bg="yellow", width=10, height=3, command=Show_Pic)
btn_show_pic.pack(side = LEFT)

### program start
btn_show_pic.config(state=DISABLED)

########## main loop
root.mainloop()