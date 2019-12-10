#from preprocessing import preprocessing_mod as pp
import numpy as np
import cv2
import skimage.io as io
from skimage.color import rgb2gray
from numba import vectorize, cuda
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.exposure import histogram
from skimage.measure import find_contours
from skimage.draw import rectangle, line
from skimage.transform import rotate

def returncell(row,col,Xlines,Ylines,binimage):
    uppery=Ylines[col]
    downy=Ylines[col+1]
    leftx=Xlines[row]
    rightx=Xlines[row+1]
    return binimage[leftx:rightx,uppery:downy]



def detectHlines(binimage):
    Xlines=[]   
    onesmax=binimage.shape[1]
    iprev=1000000
    for i in range(binimage.shape[0]):
        sliceimage=binimage[i,:]        
        sumones=np.sum(sliceimage)        
        if onesmax-sumones<5 and np.abs(i-iprev)>7:
            Xlines.append(i)
            iprev=i        
    return Xlines

def detectVlines(binimage):
    ylines=[]
    onesmax=binimage.shape[0]
    iprev=1000000
    for i in range(binimage.shape[1]):
        sliceimage=binimage[:,i]        
        sumones=np.sum(sliceimage )
        if onesmax-sumones<5 and np.abs(i-iprev)>7:
            ylines.append(i)
            iprev=i
    return ylines
        


# Read the image
img = io.imread('excelimage.jpg')
grayimag=rgb2gray(img)*255
img_bin=grayimag<250
img_bin=img_bin.astype(int)
io.imshow(img_bin,cmap='gray')
Xlines=detectHlines(img_bin)   
Ylines=detectVlines(img_bin)  
 
nameimage=returncell(10,0,Xlines,Ylines,img)
io.imshow(nameimage) 
    
    
