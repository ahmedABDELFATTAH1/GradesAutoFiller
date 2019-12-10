
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
from skimage.filters import threshold_local

def averageFilter(image, averaged_filter):
    
    filter_dimention = int(averaged_filter.shape[0])
    half_filter_dimention = int(np.floor(filter_dimention/2))
    image_dimention = image.shape[0]
    image_dimention_y = image.shape[1]    
    average_image = np.array(image,np.float64)
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
    filter_dimentionx = int(erosion_filter.shape[0])# Read the image    
    filter_dimentiony = int(erosion_filter.shape[1])# Read the image    
    half_filter_dimentionx = int(np.floor(filter_dimentionx/2))
    half_filter_dimentiony = int(np.floor(filter_dimentiony/2))
    image_dimentionx = image.shape[0]
    image_dimention_y = image.shape[1]
    erosed_image = np.zeros((image_dimentionx, image_dimention_y))
    for i in range(half_filter_dimentionx, image_dimentionx-half_filter_dimentionx):
        for j in range(half_filter_dimentiony, image_dimention_y-half_filter_dimentiony):
            result = 1
            for l in range(filter_dimentionx):
                if result == 0:
                    break
                for w in range(filter_dimentiony):
                    if erosion_filter[l][w] == 1 and image[i+l-half_filter_dimentionx][j+w-half_filter_dimentiony] == 0:
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
    filter_dimentionx = int(difilter.shape[0])
    filter_dimentiony = int(difilter.shape[1])
    half_filter_dimentionx = int(np.floor(filter_dimentionx/2))
    half_filter_dimentiony = int(np.floor(filter_dimentiony/2))
    image_dimention_x = image.shape[0]
    image_dimention_y = image.shape[1]
    dilated_image = np.zeros((image_dimention_x, image_dimention_y))
    for i in range(half_filter_dimentionx, image_dimention_x-half_filter_dimentionx):
        for j in range(half_filter_dimentiony, image_dimention_y-half_filter_dimentiony):
            result = 0
            for l in range(filter_dimentionx):
                if result == 1:
                    break
                for w in range(filter_dimentiony):
                    if difilter[l][w] == 1 and image[i+l-half_filter_dimentionx][j+w-half_filter_dimentiony] == 1:
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
    filter_dimentionx = int(edgefilter.shape[0])
    filter_dimentiony = int(edgefilter.shape[1])
    half_filter_dimentionx = int(np.floor(filter_dimentionx/2))
    half_filter_dimentiony = int(np.floor(filter_dimentiony/2))
    image_dimention = image.shape[0]
    image_dimention_y = image.shape[1]
    edgedimage = np.zeros((image_dimention, image_dimention_y))
    for i in range(half_filter_dimentionx, image_dimention-half_filter_dimentionx):
        for j in range(half_filter_dimentiony, image_dimention_y-half_filter_dimentiony):
            matresult = np.multiply(image[i-half_filter_dimentionx:i-half_filter_dimentionx +
                                          filter_dimentionx, j-half_filter_dimentiony:j-half_filter_dimentiony+filter_dimentiony], edgefilter)
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
    median_image = np.array(image,np.float64)
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
    xmax1=0
    indx1=0
    xmax2=0
    indx2=0    
    ymax1=0
    indy1=0
    ymax2=0
    indy2=0
    for i in range(image.shape[1],1,-1):
        imageslice=image[:,i-1:i]
        imageslice=imageslice.reshape(image.shape[0]*1,1)
        slicecount=np.sum(imageslice)        
        if(slicecount>xmax1):
            xmax1=slicecount
            indx1=i       
        
    for i in range(0,image.shape[1]-1,1):
        imageslice=image[:,i:i+1]
        imageslice=imageslice.reshape(image.shape[0],1)
        slicecount=np.sum(imageslice)
        if(slicecount>xmax2 and slicecount<xmax1 and np.abs(i-indx1)>10):
            indx2=i
            xmax2=slicecount
            
    for i in range(image.shape[0],1,-1):
        imageslice=image[i-1:i,:]
        imageslice=imageslice.reshape(image.shape[1]*1,1)
        slicecount=np.sum(imageslice)
        if(slicecount>ymax1):
            ymax1=slicecount
            indy1=i       
    for i in range(0,image.shape[0]-1,1):
        imageslice=image[i:i+1,:]
        imageslice=imageslice.reshape(image.shape[1]*1,1)
        slicecount=np.sum(imageslice)        
        if(slicecount>ymax2 and slicecount<ymax1 and np.abs(i-indy1)>10):            
            indy2=i
            ymax2=slicecount
    minx=min(indx1,indx2)
    maxx=max(indx1,indx2) 
    miny=min(indy1,indy2)
    maxy=max(indy1,indy2)     
    original=original[int(miny*8):int(maxy*8),int(minx*8):int(maxx*8),:]
    return original
               
   
    




def helpfunc(H,mystart,myend):    
    CH=np.zeros(myend-mystart)
    CH[0]=H[mystart]
    for i in range(len(CH)):
        CH[i]=CH[i-1]+H[i+mystart]
    mysum=0
    thr=0
    for i in range(len(CH)):
        mysum=mysum+(H[i+mystart]*(i+mystart))
    thr=np.round(mysum/CH[-1])    
    return thr.astype('uint8')


def getthreshold(image):    
    image=image.astype('uint8')
    H=np.zeros(256)   
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            H[image[i,j]]=H[image[i,j]]+1
    H=H.astype('uint8')
    thr=helpfunc(H,0,len(H))  
      
    newthr=-1
    while(1):
        if(thr!=0 or thr!=256):
            thr1=helpfunc(H,0,thr)
            thr2=helpfunc(H,thr,256)  
            newthr=int((thr1.astype(int)+thr2.astype(int))/2)
        else:
            newthr=thr
        if newthr == thr :
            break
        else :
            thr=newthr 
    return thr
 
def imageEnhancement(image,graylevels):         
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


def detectHlines(binimage):
    Xlines=[]       
    iprev=1000000
    for i in range(binimage.shape[0]):
        sliceimage=binimage[i,:]        
        sumones=np.sum(sliceimage)        
        if sumones>200 and np.abs(i-iprev)>4:
            Xlines.append(i)
            iprev=i        
    return Xlines


def detectVlines(binimage):
    ylines=[]    
    iprev=1000000
    for i in range(binimage.shape[1]):
        sliceimage=binimage[:,i]        
        sumones=np.sum(sliceimage )
        if sumones>200 and np.abs(i-iprev)>4:
            ylines.append(i)
            iprev=i
    return ylines
 
   
 
#first load the image
colorimage = io.imread('excelpic/7.jpg')
#convert it to gray scale
image = rgb2gray(colorimage)
image = image*255  
#io.imshow(colorimage)
image=image.astype('uint8')
#edgedimage =cv2.Canny(image,150,200)
#io.imshow(edgedimage,cmap='gray')
#edgedimage=edgedimage==255
#H, angle = hughspace(edgedimage, 1)
#rotatedimg=rotate(edgedimage,90-angle)
#colorimage=rotate(colorimage,90-angle)
#io.imshow(colorimage,cmap='gray')
grayimage=rgb2gray(colorimage)*255
Egrayimage=imageEnhancement(grayimage,265)
#io.imshow(Egrayimage,cmap='gray')

def edgesfilter(image):
    edgefilterx = np.array([[1, 1,2,1,1], [0,0,0,0,0], [-1,-1,-2,-1,-1]])
    edgefiltery = np.array([[1,0,-1], [1,0,-1], [2,0,-2],[1,0,-1], [1,0,-1]])
    edgesimagex=np.abs(sobelFilter(image,edgefilterx))
    edgesimagey=np.abs(sobelFilter(image,edgefiltery))
    edgesimagex=edgesimagex>300
    edgesimagey=edgesimagey>300
    openingfiltery=np.ones((20,1))
    openingfilterx=np.ones((1,20))
    edgesimagey=opening(edgesimagey,openingfiltery)
    edgesimagex=opening(edgesimagex,openingfilterx)
    return edgesimagex,edgesimagey

from skimage.transform import hough_line, hough_line_peaks

def linepeaks(image):
    H, theta, d = hough_line(image)
    _,angles,distance=hough_line_peaks(H,theta,d)
    return angles,distance
    

def intersection(line1,line2,edgeimage):
    pt=[]
    for i in range(edgeimage.shape[0]):
        for j in range(edgeimage.shape[1]):
            radiusx=(line1[0]*np.pi)/180
            radiusy=(line2[0]*np.pi)/180                    
            if(np.abs(line1[1]-(j*np.cos(radiusx)+(i*np.sin(radiusx))))<.5 and np.abs(line2[1]-(j*np.cos(radiusy)+(i*np.sin(radiusy))))<.5):
                pt=[i,j]
    return pt

def returncell(row,col,Xlines,Ylines,binimage):    
    upperx=int(np.round(Xlines[row]))
    downx=int(np.round(Xlines[row+1]))
    lefty=int(np.round((Ylines[col]/(np.cos(diction[Ylines[col]])))-(Xlines[row]*np.tan(diction[Ylines[col]]))))
    righty=int(np.round((Ylines[col+1]/(np.cos(diction[Ylines[col+1]])))-(Xlines[row+1]*np.tan(diction[Ylines[col+1]]))))
    print(lefty,righty)
    return binimage[upperx:downx,lefty:righty,:]  


edgesimagex,edgesimagey=edgesfilter(Egrayimage)
io.imshow(edgesimagex,cmap='gray')
io.imshow(edgesimagey,cmap='gray')


Xangle,Xdistance=linepeaks(edgesimagex)
Xdistances=np.sort(Xdistance)
Yangle,Ydistance=linepeaks(edgesimagey)
Ydistances=np.sort(Ydistance)
diction={}
for i in range(len(Ydistance)):
    diction[Ydistance[i]]=Yangle[i]

cell=returncell(9,3,Xdistances,Ydistances,colorimage)

io.imshow(cell,cmap='gray')


t_leftpt=intersection(linex1,liney1,edgesimagex)
t_rightpt=intersection(linex1,liney2,edgesimagex)
b_leftpt=intersection(linex2,liney1,edgesimagex)
b_rightpt=intersection(linex2,liney2,edgesimagex)

pts1 = np.float32(
    [t_leftpt,t_rightpt,b_rightpt,b_leftpt]
)
pts1[0]
rows, cols, ch = rows, cols, ch = colorimage.shape 
pts2 = np.float32(
    [[0,0],[0,300],[300,300],[300,0]]
)    




M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(colorimage, M, (cols, rows))




cellimg=returncell(10,6,Xlines,Ylines,colorimage)
#io.imshow(cellimg)






# =============================================================================
# colorimage = #io.imread('excelpic/11.jpg')
# image = rgb2gray(colorimage)
# image=image*255
# mfilter=np.ones((3,3))
# newimg=MedianFilter(image,mfilter)
# newimg=newimg.astype('uint8')
# #io.imshow(image,cmap='gray')
# #io.imshow(newimg,cmap='gray')
# edges = cv2.Canny(newimg,150,200)
# 
# 
# #io.imshow(edges,cmap='gray')
# edges=edges.astype('uint8')  
# from skimage.transform import hough_line, hough_line_peaks
# tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
# h, theta, d = hough_line(edges, theta=tested_angles)
# mymax=0
# maxangle=0
# for i in range(h.shape[0]):
#     for j in range(h.shape[1]):
#         if(h[i][j]>mymax):
#             mymax=h[i][j]
#             maxangle=j
# newh=np.zeros((1,360))
# 
# mylist=[]
# for i in range(h.shape[0]):
#     for j in range(h.shape[1]):
#         if h[i][j]>10:            
#             if(mymax/h[i][j]>1.3 and mymax/h[i][j]<1.5):
#                 mylist.append(j-180)
# 
# 
# 
# mysum=0
# for i in range(h.shape[1]):
#     for j in range(h.shape[0]):
#         if(h[j][i]>mymax/2):
#             mysum+=1
#     newh[0,i]=mysum
#     mysum=0
#         
#        
#         
#         
#         
# HK,angle,dist=hough_line_peaks(h, theta, d)
# #io.imshow(h,cmap='gray')
# maxangle=angle*(180/np.pi)
# 
# 
# =============================================================================
