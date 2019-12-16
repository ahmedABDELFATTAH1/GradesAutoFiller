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
    rowscells=excelpreprocessing('image/333.jpg')
    for row in rowscells:
        for cellinfo in row:
            cell=cellinfo[0]
            cell=cv2.bitwise_not(cell)
            kernalx=np.ones((3,3))
            erosion = cv2.erode(cell,kernalx,iterations = 1)
            cell=cv2.bitwise_not(cell)
            cv2.imshow('img',erosion)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
            
    
    
    

    
