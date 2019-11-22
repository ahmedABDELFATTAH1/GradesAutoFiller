from preprocessing import preprocessing_mod as pp
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

orimage=io.imread('testimgs/stress/1.jpg')
io.imshow(orimage)
image=pp.preprocessimage('testimgs/stress/2.jpg')
io.imshow(image)