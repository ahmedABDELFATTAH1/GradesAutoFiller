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
import imutils

def excelpreprocessing(path):
    colorimage= cv2.imread(path)
    gray_scale_image=cv2.cvtColor(colorimage,cv2.COLOR_BGR2GRAY)   
    edge_gray_scale_image =cv2.Canny(gray_scale_image,100,150)
    width = 720
    height = 960
    ratiowidth=colorimage.shape[1]/width
    ratioheight=colorimage.shape[0]/height
    if ratiowidth > 1.5 and ratioheight > 1.5:
        dim = (width, height)    
        resized = cv2.resize(gray_scale_image, dim, interpolation = cv2.INTER_NEAREST)
        edge_gray_scale_image =cv2.Canny(resized,100,150)
        print(ratiowidth,ratioheight)    
    th2 = cv2.adaptiveThreshold(gray_scale_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)         
    edge_gray_scale_image=cv2.medianBlur(edge_gray_scale_image,3)
    cv2.imshow('img',edge_gray_scale_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
           
    kernalx=np.ones((1,6))
    imagex=cv2.morphologyEx(edge_gray_scale_image, cv2.MORPH_OPEN, kernalx)
    kernalx=np.ones((1,20))
    imagex=cv2.morphologyEx(imagex, cv2.MORPH_CLOSE, kernalx)
    
    
    cv2.imshow('img',imagex)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    kernaly=np.ones((7,1))
    imagey=cv2.morphologyEx(edge_gray_scale_image, cv2.MORPH_OPEN, kernaly)
    kernaly=np.ones((20,1))
    imagey=cv2.morphologyEx(imagey, cv2.MORPH_CLOSE, kernaly)
    
    
    cv2.imshow('img',imagey)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    binaryimage = cv2.bitwise_or(imagey, imagex, mask = None) 
    
  
    cv2.imshow('img',binaryimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    lines = cv2.HoughLines(binaryimage, 1, np.pi / 180, 150)
    
    newlines=[]
    for line in lines:
        rho,theta = line[0]
        if rho <0 and theta<0:
            rho=np.abs(rho) 
            theta=theta+np.pi
        newlines.append([rho,theta])
        
    import operator
    newlines = sorted(newlines, key=operator.itemgetter(0, 1))
    
    for i in range(len(newlines)-1):
        if np.abs(newlines[i][0]-newlines[i+1][0])<15 and np.abs(newlines[i][1]-newlines[i+1][1])<.022:
            linerho=(newlines[i][0]+newlines[i+1][0])/2
            linetheta=(newlines[i][1]+newlines[i+1][1])/2
            newlines[i][0]=linerho
            newlines[i+1][0]=linerho
            newlines[i][1]=linetheta
            newlines[i+1][1]=linetheta
    
    
    binaryimage=cv2.bitwise_not(binaryimage)
    
    
    for line in newlines:
        rho,theta = line
        if rho <0:
            rho=np.abs(rho) 
            theta=theta+np.pi             
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # x1 stores the rounded off value of (r * cos(theta) - 1000 * sin(theta))
        x1 = int(x0 + 1000 * (-b))
        # y1 stores the rounded off value of (r * sin(theta)+ 1000 * cos(theta))
        y1 = int(y0 + 1000 * (a))
        # x2 stores the rounded off value of (r * cos(theta)+ 1000 * sin(theta))
        x2 = int(x0 - 1000 * (-b))
        # y2 stores the rounded off value of (r * sin(theta)- 1000 * cos(theta))
        y2 = int(y0 - 1000 * (a))
        cv2.line(binaryimage, (x1, y1), (x2, y2), (0, 0, 0), 2)
    
    
    
    cv2.imshow('image', binaryimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    binaryimage=cv2.bitwise_not(binaryimage)
    
    
    
    cv2.imshow('img',binaryimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
         
    binaryimage=binaryimage/255
    
    contours=find_contours(binaryimage,.8)
    
    
    allcells=[]
    for cont in contours:
        xup=int(np.min(cont[:,0]))
        xdown=int(np.max(cont[:,0]))
        yleft=int(np.min(cont[:,1]))
        yright=int(np.max(cont[:,1]))
        width=yright-yleft
        height=xdown-xup
        if(width>20 and width<60 and height >10 ):
            cell=th2[int(xup*ratioheight):int(xdown*ratioheight),int(yleft*ratiowidth):int(yright*ratiowidth)]          
            allcells.append([cell,xup,yleft])
    rowscells=[]
    index=-1
    prev_xup=-10000
    for cellinfo in allcells:
        xup=cellinfo[1]
        if xup-prev_xup>3:
            rowscells.append([cellinfo])      
            prev_xup=xup
            index+=1
        else:
            rowscells[index].append(cellinfo)
            prev_xup=xup    
    
    newrowscells=[]  
    index=0
    for row in rowscells:        
        if len(row) < 5:
            continue              
        newrowscells.append([])
        sortedrow=sorted(row,key=lambda x: x[2],reverse=True)
        newrowscells[index]=sortedrow        
        index+=1
    return newrowscells

def extraxctidname(path):
    colorimage= cv2.imread(path)
    gray_scale_image=cv2.cvtColor(colorimage,cv2.COLOR_BGR2GRAY)
    edge_gray_scale_image =cv2.Canny(gray_scale_image,100,150)
    cv2.imshow('img',gray_scale_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    contours=find_contours(edge_gray_scale_image,.8)
    
    nameid=[]
    for cont in contours:
        xup=int(np.min(cont[:,0]))
        xdown=int(np.max(cont[:,0]))
        yleft=int(np.min(cont[:,1]))
        yright=int(np.max(cont[:,1]))
        width=yright-yleft
        height=xdown-xup
        if width/height>6 and width/height <6.5:
            print(width/height)
            nameid.append([xup,xdown,yleft,yright])
    
    if nameid[0][2] < nameid[1][2]:
        return gray_scale_image[nameid[0][0]:nameid[0][1],nameid[0][2]:nameid[0][3]],gray_scale_image[nameid[1][0]:nameid[1][1],nameid[1][2]:nameid[1][3]]
    else:
        return gray_scale_image[nameid[1][0]:nameid[1][1],nameid[1][2]:nameid[1][3]],gray_scale_image[nameid[0][0]:nameid[0][1],nameid[0][2]:nameid[0][3]]
        

            


