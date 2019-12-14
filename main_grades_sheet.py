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
from preprocessing.excellpre import *

rowscells=preprocessing('excelpic/8.jpg')

cv2.imshow('img',rowscells[5][7])
cv2.waitKey(0)
cv2.destroyAllWindows()

    
