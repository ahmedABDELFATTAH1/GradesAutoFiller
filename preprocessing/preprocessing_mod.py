
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


def averageFilter(image, averaged_filter):
    
    filter_dimention = int(averaged_filter.shape[0])
    half_filter_dimention = int(np.floor(filter_dimention/2))
    image_dimention = image.shape[0]
    image_dimention_y = image.shape[1]
    average_image = np.random.rand(image_dimention, image_dimention_y)*255
    for i in range(half_filter_dimention, image_dimention-half_filter_dimention):
        for j in range(half_filter_dimention, image_dimention_y-half_filter_dimention):
            matresult = np.multiply(image[i-half_filter_dimention:i-half_filter_dimention +
                                          filter_dimention, j-half_filter_dimention:j-half_filter_dimention+filter_dimention], averaged_filter)
            result = np.sum(matresult)
            average_image[i][j] = result
    return average_image

'''
erosion -> this function perform erosion operation on a given binary image 

parameters : image :- the image to be diated 
             erosion_filter:filter used for erosion 

return : erosed_image:- the image after erosion
'''

def erosion(image, erosion_filter):
    filter_dimention = int(erosion_filter.shape[0])
    half_filter_dimention = int(np.floor(filter_dimention/2))
    image_dimention = image.shape[0]
    image_dimention_y = image.shape[1]
    erosed_image = np.zeros((image_dimention, image_dimention_y))
    for i in range(half_filter_dimention, image_dimention-half_filter_dimention):
        for j in range(half_filter_dimention, image_dimention_y-half_filter_dimention):
            result = 1
            for l in range(filter_dimention):
                if result == 0:
                    break
                for w in range(filter_dimention):
                    if erosion_filter[l][w] == 1 and image[i+l-half_filter_dimention][j+w-half_filter_dimention] == 0:
                        result = 0
                        break
            erosed_image[i][j] = result
    return erosed_image

'''
dilation -> this function perform dilation operation on a given binary image 

parameters : image :- the image to be diated 
             difilter:filter used for dilation 


return : newimage:- the image after dilation
'''
def dilation(image, difilter):
    filter_dimention = int(difilter.shape[0])
    half_filter_dimention = int(np.floor(filter_dimention/2))
    image_dimention_x = image.shape[0]
    image_dimention_y = image.shape[1]
    dilated_image = np.zeros((image_dimention_x, image_dimention_y))
    for i in range(half_filter_dimention, image_dimention_x-half_filter_dimention):
        for j in range(half_filter_dimention, image_dimention_y-half_filter_dimention):
            result = 0
            for l in range(filter_dimention):
                if result == 1:
                    break
                for w in range(filter_dimention):
                    if difilter[l][w] == 1 and image[i+l-half_filter_dimention][j+w-half_filter_dimention] == 1:
                        result = 1
                        break
            dilated_image[i][j] = result
    return dilated_image




def closing(image, closefilter):
    image = dilation(image, closefilter)
    image = erosion(image, closefilter)
    return image


def opening(image, openfilter):
    image = erosion(image, openfilter)
    image = dilation(image, openfilter)
    return image


def sobelFilter(image, edgefilter):
    filter_dimention = int(edgefilter.shape[0])
    half_filter_dimention = int(np.floor(filter_dimention/2))
    image_dimention = image.shape[0]
    image_dimention_y = image.shape[1]
    edgedimage = np.zeros((image_dimention, image_dimention_y))
    for i in range(half_filter_dimention, image_dimention-half_filter_dimention):
        for j in range(half_filter_dimention, image_dimention_y-half_filter_dimention):
            matresult = np.multiply(image[i-half_filter_dimention:i-half_filter_dimention +
                                          filter_dimention, j-half_filter_dimention:j-half_filter_dimention+filter_dimention], edgefilter)
            result = np.sum(matresult)
            edgedimage[i][j] = result
    return edgedimage


def applysobeelfilter(image):
    edgefilterx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    edgefiltery = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    imagx = sobelFilter(image, edgefilterx)
    imagy = sobelFilter(image, edgefiltery)
    edgedimage = np.zeros((image.shape[0], image.shape[1]))
    edgedimage = np.sqrt(np.power(imagx, 2)+np.power(imagy, 2))
    return edgedimage


def applythreshold(image, threshold):
    binaryimg = image > threshold
    binaryimg = binaryimg.astype(int)
    return binaryimg

def MedianFilter(image, median_filter):
    filter_dimention = int(median_filter.shape[0])
    half_filter_dimention = int(np.floor(filter_dimention/2))
    image_dimention = image.shape[0]
    image_dimention_y = image.shape[1]
    median_image = np.zeros((image_dimention, image_dimention_y))
    for i in range(half_filter_dimention, image_dimention-half_filter_dimention):
        for j in range(half_filter_dimention, image_dimention_y-half_filter_dimention):
            result = []
            matresult = np.copy(
                image[i-half_filter_dimention:i-half_filter_dimention+filter_dimention, j-half_filter_dimention:j-half_filter_dimention+filter_dimention])
            result = sorted(np.reshape(matresult, filter_dimention*filter_dimention))
            median_image[i][j] = result[int(len(result)/2)]
    return median_image


def thresholding(image,threshold):
    return (image>threshold).astype(int)


def cropppimg(image,original):    
    for i in range(image.shape[1]-1,-1,-3):
        imageslice=image[:,i-3:i]
        imageslice=imageslice.reshape(image.shape[0]*3,1)
        slicecount=np.sum(imageslice)
        if(slicecount/(image.shape[1]*3)>.3):
            original=original[:,:int(np.abs((i-3)*8)),:]
            break    
    for i in range(0,image.shape[1],3):
        imageslice=image[:,i:i+3]
        imageslice=imageslice.reshape(image.shape[0]*3,1)
        slicecount=np.sum(imageslice)
        if(slicecount/(image.shape[1]*3)>.3):
            original=original[:,int(np.abs((i+3)*8)):,:]
            break
    for i in range(image.shape[0],3,-3):
        imageslice=image[i-3:i,:]
        imageslice=imageslice.reshape(image.shape[1]*3,1)
        slicecount=np.sum(imageslice)
        if(slicecount/(image.shape[1]*3)>.3):
            original=original[:int(np.abs((i-3)*8)),:,:]
            break
    
    for i in range(0,image.shape[0]-3,3):
        imageslice=image[i:i+3,:]
        imageslice=imageslice.reshape(image.shape[1]*3,1)
        slicecount=np.sum(imageslice)
        
        if(slicecount/(image.shape[1]*3)>.3):
            original=original[int(np.abs((i+3)*8)):,:]
            break 
     
    
    return original
               
        


def hughspace(image, angle_resolution):
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_resolution))
    y_length, x_length = image.shape
    max_distance = int(np.ceil(np.sqrt(y_length*y_length + x_length*x_length)))
    number_of_thetas = len(thetas)
    hough_space = np.zeros((2*max_distance, number_of_thetas))
    max_bin_size = 0
    angle = 0
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    for y in range(y_length):
        for x in range(x_length):
            if image[y][x] == 1:
                for i_theta in range(number_of_thetas):
                    pho = int(
                        np.ceil(x*cos_thetas[i_theta] + y*sin_thetas[i_theta]))
                    hough_space[max_distance + pho][i_theta] += 1
                    if(hough_space[max_distance + pho][i_theta] > max_bin_size):
                        angle = i_theta
                        max_bin_size = hough_space[max_distance + pho][i_theta]
        
    for pho in range(hough_space.shape[0]):
        for theta in range(hough_space.shape[1]):
            hough_space[pho][theta] *= (255.0/max_bin_size)
    return hough_space, angle



def preprocessimage(path):   
   #first load the image
    colorimage = io.imread(path)
    #then resize to make it faster to process
    resizedwidth = int(colorimage.shape[0]/8)
    resizedhigh = int(colorimage.shape[1]/8)
    colorimageresized = cv2.resize(colorimage, dsize=(
        resizedhigh, resizedwidth), interpolation=cv2.INTER_LINEAR)
    #convert it to gray scale
    image = rgb2gray(colorimageresized)
    image = image*255
    myfilter = 1/25*(np.ones((5, 5)))
    medianfilter = np.ones((5, 5))   
    image = MedianFilter(image, medianfilter)    
    image = averageFilter(image, myfilter)   
    edgedimage = applysobeelfilter(image)   
    binaryimage = applythreshold(edgedimage, 100)    
    closingfilter = np.ones((3, 3))
    binaryimage = opening(binaryimage, closingfilter)    
    H, angle = hughspace(binaryimage, 1)
    rotatedimg=rotate(binaryimage,angle-90)
    colorimage=rotate(colorimage,angle-90)    
    rotatedimg=applythreshold(rotatedimg,.5)
    original=cropppimg(rotatedimg,colorimage)
    return original
    
    

