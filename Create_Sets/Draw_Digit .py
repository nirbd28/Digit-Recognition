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
import cv2
########################################

######################################## Import from folder
folder_name='Functions'
path_name_new=path_main + '\\' + folder_name
sys.path.append(path_name_new)
##### Import Files
import Main_Func, Define
########################################

################################################################################
################################################################################
################################################################################

########## get digit to draw
digit_input = input('Digit to draw: ')

########## Instructions
print('--------------------')
print('Instructions:')
print('Right Click - clear and save picture')
print('Double Left Click - exit program and save set')

########## inputs
pixel_window=Define.Parameters.pixel_window
pixel_norm=Define.Parameters.pixel_norm
draw_radius=Define.Parameters.draw_radius

##########
def Concatenate_Image(features, pic):
    pic = np.expand_dims(pic, axis=0)
    features = np.concatenate((features,pic), axis = 0) 
    return features
##########

############################################################ mouse callback function

### parameters
img=[]
features=[]
counter=0 
flag_exit=0
###
drawing = False 

##########
def draw_circle(event,x,y,flags,param):
    global drawing, img, features
    global counter, flag_exit

    if event == cv2.EVENT_LBUTTONDBLCLK: # double click for exit and save data 
        flag_exit=1

    if event == cv2.EVENT_RBUTTONDOWN: #  click for save img and clear
		
        ### picture proccesing
        pic=Main_Func.Picture_Proccess(img, pixel_norm)

        ### append to features
        if (counter==0):
            pic = np.expand_dims(pic, axis=0)
            features = pic 
        else:
            features=Concatenate_Image(features, pic)

        ###    
        counter+=1
        print('-----')
        print('***',counter)
        ### new window
        img=np.full((pixel_window,pixel_window,3),0,np.float64) # new window
      
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),draw_radius,(255,255,255),-1)
                
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    
############################################################
    
img=np.full((pixel_window,pixel_window,3),0,np.float64) # new window

### draw loop
while(flag_exit==0): 
    ### show window
    cv2.imshow('Draw Digit',img)
    cv2.namedWindow('Draw Digit')
    cv2.setMouseCallback('Draw Digit',draw_circle) # call draw function

    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()

#################### create labels
labels=np.zeros(counter)
labels.fill(int(digit_input))
labels = labels.astype(np.uint8)

########## pack set
set=Main_Func.Pack_Set(features, labels)

#################### save to file
file_name=path_Digits_Data+'\\'+digit_input +'.obj'
filehandler = open(file_name,"wb")
pickle.dump(set,filehandler)
filehandler.close()
#
print('File', digit_input, 'was saved', 'in folder Digits_Data')

