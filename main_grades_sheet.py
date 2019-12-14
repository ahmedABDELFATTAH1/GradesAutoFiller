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
from preprocessing.preprocessing_mod import *



if __name__=='__main__':        
    grayimage=deskewImage('excelpic/16.jpg') 
    th3 = cv2.adaptiveThreshold(grayimage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    X_lines,Y_lines=preprocessing(grayimage)
    cell=returncell(32,5,X_lines,Y_lines,th3)
    newcell=cv2.medianBlur(cell,3)
    cv2.imshow('img',newcell)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    
