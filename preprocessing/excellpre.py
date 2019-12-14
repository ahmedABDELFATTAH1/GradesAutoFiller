import numpy as np
import cv2
import skimage.io as io
from skimage.color import rgb2gray
from numba import vectorize, cuda
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.exposure import histogram
from skimage.measure import find_contours
#from skimage.draw import rectangle, line
from skimage.transform import rotate
from skimage.filters import threshold_local,median
from skimage.transform import hough_line, hough_line_peaks
from skimage.morphology import skeletonize

def intersection(line1, line2):  
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def returncell(row,col,Xlines,Ylines,binimage):
    uppery=Ylines[col]
    downy=Ylines[col+1]
    leftx=Xlines[row]
    rightx=Xlines[row+1]
    return binimage[leftx:rightx,uppery:downy]



def detectHlines(binimage):
    Xlines=[]  
    iprev=1000000
    for i in range(0,binimage.shape[0],1):
        sliceimage=binimage[i,:]        
        sumones=np.sum(sliceimage)        
        if sumones>20000 and np.abs(i-iprev)>15:
            Xlines.append(i+5)
            iprev=i        
    return Xlines

def detectVlines(binimage):
    ylines=[]
    iprev=1000000
    for i in range(0,binimage.shape[1]-2,2):
        sliceimage=binimage[:,i:i+2]        
        sumones=np.sum(sliceimage )
        if sumones>50000 and np.abs(i-iprev)>10:
            ylines.append(i+5)
            iprev=i
    return ylines



def preprocessing(gray_scale_image):      
    edge_gray_scale_image =cv2.Canny(gray_scale_image,100,150)
    
    kernelx = np.ones((1,8))    
    openingx = cv2.morphologyEx(edge_gray_scale_image, cv2.MORPH_OPEN, kernelx)
    
    kernely = np.ones((8,1)) 
    openingy = cv2.morphologyEx(edge_gray_scale_image, cv2.MORPH_OPEN, kernely)
    
    img1_bg = cv2.bitwise_or(openingx,openingy)  
    
    X_lines=detectHlines(openingx)
    Y_lines=detectVlines(openingy)
    
    return X_lines,Y_lines
    
    
    
        


    
    
    

