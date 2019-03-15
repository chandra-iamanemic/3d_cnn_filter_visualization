# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 11:25:48 2019

@author: cks
"""


#Import the required libraries
import numpy as np
import time

#Define a function to perform a 3D convolution using a 3D filter on a set of stacked image frames
def my_3d_conv(input_layer, filters):
    
    #Determine the dimensions from the input (height, width, channels and the number of frames that are stacked)
    (frames,height,width,channels) = input_layer.shape
    
    #Determine the dimensions of the filter that is used to convolve the input volume
    f_h,f_w,f_f,channels = filters.shape
    
    #From the obtained values of filter and input dimensions determine the output feature volume dimensions
    n_h = (height-f_h) + 1
    n_w = (width-f_w)+1
    n_f = (frames-f_f) + 1
    
    #Initiate the feature volume using a zero matrix (np.zeros)
    feature_volume = np.zeros((n_h,n_w,n_f))
    
    
    #Iterate through the feature volume and determine the value at each space by convolving the filter with the selected slice of volume
    for i in range(n_h):
        for j in range(n_w):
            for k in range(n_f):
                
                feature_volume[i,j,k] = np.sum(np.multiply(input_layer[k:k+f_h,i:i+f_w,j:j+f_f,:],filters))
                
                
                
                
    return feature_volume

import cv2



# Red the 16 frames from the folder into a list
import glob


#Read the images from the file into a list
images = [cv2.imread(file) for file in glob.glob("C:/Users/cks/Documents/practice codes/frames/*.jpg")]

#Initiate the input layer 
input_layer = np.zeros((16,256,256,3),dtype = np.uint8)

#Resize the 16 image frames from the list into 256x256
for i in range(16):
    
    images[i] = cv2.resize(images[i],(256,256))
    

#Copy the images in the list into our input_layer matrix which can be fed into our 3D Conv function
for i in range(16):

    input_layer[i] = images[i]    


#Run the images in loop and display them frame by frame to visualize the input given to the 3D Conv function    
for i in range(16):
    
    cv2.imshow("frames",input_layer[i,:,:,:])
    cv2.waitKey(0)

cv2.destroyAllWindows()

#Create a random volume of values which we can use as our filter/kernel
np.random.seed(seed=5)
filters = 0.5*np.random.randn(5,5,5,3)                

#Running the 3d convolution on our input data of 16 frames using a random 4x4x4 3 channel kernel

feature_volume = my_3d_conv(input_layer,filters)

#Determine the maximum and minimum value from our feature volume so that we can normalize it 
max_mat = np.amax(feature_volume)
min_mat = np.amin(feature_volume)

feature_volume = feature_volume/max_mat


#Determine the shape of our feature volume
(out_h,out_w,out_f) = feature_volume.shape

#Run the feature volume frame by frame and view the feature volume as images to visualize the features that our random kernel has captured 
for i in range(out_f):

    
    cv2.imshow("z",feature_volume[:,:,i])
    cv2.waitKey(1)
    time.sleep(0.5)
    
cv2.destroyAllWindows()


#
## Trying on the dot data
#
#import pickle
#
#
##Load the matrix X from the pickled file
#
#X = pickle.load(open("dot_data_64.pickle","rb"))
#Y = pickle.load(open("dot_labels_64.pickle","rb"))
#X_test = pickle.load(open("dot_test_data_64.pickle","rb"))
#Y_test = pickle.load(open("dot_test_labels_64.pickle","rb"))
#
#
#dot_input = X.reshape(150,16,64,64,1)
#
#feature_volume_dot = my_3d_conv(dot_input[0,:,:,:,:],filters)
#
#
#for i in range(12):
#
#    
#    cv2.imshow("z",feature_volume_dot[:,:,i])
#    cv2.waitKey(1)
#    time.sleep(0.5)
#    
#cv2.destroyAllWindows()

    
