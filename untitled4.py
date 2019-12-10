from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from skimage.draw import rectangle
from skimage.color import rgb2gray,gray2rgb
from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from skimage.draw import ellipse

import numpy as np
import cv2
import skimage.io as io



def getImageWithHist(image,graylevels):         
    size=image.shape[0]*image.shape[1]
    H=np.zeros(graylevels)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):            
            H[int(image[i,j])]=H[int(image[i,j])]+1   
    H=np.array(H)   
    H_c=np.zeros(len(H))
    H_c[0]=H[0]
    for i in range(1,len(H)):
        H_c[i]=H_c[i-1]+H[i]    
    q=np.round(((graylevels-1)*H_c)/size)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):            
            image[i,j]=q[int(image[i,j])]   
    return image


colorimage = io.imread('excelpic/1.jpg')
#io.imshow(colorimage) 
image = rgb2gray(colorimage)
image=image*255
image=getImageWithHist(image,256)
binaryimage=image < 50

coords = corner_peaks(corner_harris(binaryimage), min_distance=50)
coords_subpix = corner_subpix(binaryimage, coords, window_size=13)

fig, ax = plt.subplots()
ax.imshow(binaryimage, cmap=plt.cm.gray)

ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
        linestyle='None', markersize=6)
ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
ax.axis((0, 310, 200, 0))
plt.show()