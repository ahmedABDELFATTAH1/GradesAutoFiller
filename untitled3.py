import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from skimage.draw import rectangle
from skimage.color import rgb2gray,gray2rgb

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
#io.imshow(binaryimage,cmap='gray') 



contours = find_contours(binaryimage, .8)

for contour in contours:
    image = drawShape(image, contour, [255, 0, 0]) 

def drawShape(img, coordinates, color):
    # In order to draw our line in red
    img = gray2rgb(img)

    # Make sure the coordinates are expressed as integers
    coordinates = coordinates.astype(int)

    img[coordinates[:, 0], coordinates[:, 1]] = color

    return img



io.imshow(image,cmap='gray')



















bounding_boxes=find_contours(binaryimage, 0.8)

for box in bounding_boxes:
    [Xmin, Xmax, Ymin, Ymax] = 
    rr, cc = rectangle(start = (Ymin,Xmin), end = (Ymax,Xmax), shape=img_gray.shape)
    image[rr, cc] = 1 #set color white





rows,cols,ch = img.shape

pts1 = np.float32([[74,37],[679,8],[711,909]])
pts2 = np.float32([[10,37],[679,8],[679,909]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

