import numpy as np
import cv2
import skimage.io as io
from skimage.color import rgb2gray
from numba import vectorize, cuda
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.exposure import histogram
from skimage.measure import find_contours
from skimage.transform import rotate
from skimage.filters import threshold_local,median
from skimage.transform import hough_line, hough_line_peaks
from preprocessing.excellpre import preprocessing,returncell
from skimage.morphology import skeletonize


colorimage= cv2.imread('msq/3.jpg')
gray_scale_image=cv2.cvtColor(colorimage,cv2.COLOR_BGR2GRAY)
edge_gray_scale_image =cv2.Canny(gray_scale_image,100,150)   

     
cv2.imshow('img',edge_gray_scale_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

edge_gray_scale_image=edge_gray_scale_image/255

    
    
    
    
    