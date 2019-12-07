from commonfunctions import *
import numpy as np
import skimage.io as io
from skimage.transform import rotate
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import binary_erosion, binary_dilation, binary_closing,skeletonize, thin,erosion,dilation
from skimage.measure import find_contours
from skimage.draw import rectangle

# Show the figures / plots inside the notebook


def digits_loc(img,X=28,Y=28):
    """
    Localization of the digits , it separates each digit into a fixed size output
    Arugments :
    -- img : numpy array
    Returns
    -- digits :  Array of fixed size matrices for each digit .
    """
    X = int(X/2)
    Y = int(Y/2)
    img = rgb2gray(img) 
    img = rotate(img,270,resize=True)
    img_hist = histogram(img, nbins=2)
    # Checking the image`s background must be black and digits be white
    # Negative Transformation in case of white (objects) is more than black (background)
    if ( img_hist[0][0] < img_hist[0][1] ):
        img = 1 - img 
    
    digits = []
    # Find contours for each digit has its own contour
    contours = find_contours(img, 0.4,fully_connected='high',positive_orientation='high')
    for n, contour in enumerate(contours):
        Ymax = np.amax(contour[:, 0])
        Ymin = np.amin(contour[:, 0])
        Xmax = np.amax(contour[:, 1])
        Xmin = np.amin(contour[:, 1])
        digit_seg = ([img[int(Ymin): int(Ymax)+1, int(Xmin): int(Xmax)+1]])
        digit = np.zeros([X*2,Y*2])
        h,w = np.array(digit_seg[0]).shape
        if(h > 28 or w>28):
            continue
        digit[X-int((h+1)/2):X+int(h/2) ,Y-int((w+1)/2):Y+int(w/2) ,  ] = digit_seg[0]
        digit = rotate(digit,90,resize=True)
        digit = erosion(digit)
        digit = dilation(digit)
        digits.append(digit)
        
    return digits