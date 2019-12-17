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
'''
name : excelpreprocessing

given a path to an excel sheet  image return all cells in good formated way

'''
def excelpreprocessing(path,args):
    if args==0:  
        colorimage= cv2.imread(path)#read the file
        gray_scale_image=cv2.cvtColor(colorimage,cv2.COLOR_BGR2GRAY)  #convert it to gray scale
    elif args==1 or args==2:
        gray_scale_image=rotation(path,args)
    elif args==3:
        gray_scale_image=mcqpreprocessing(path)
        
    edge_gray_scale_image =cv2.Canny(gray_scale_image,100,150) #canny edge detection
    width = 720 #perfect width to our application
    height = 960 #perfect height 
    ratiowidth=gray_scale_image.shape[1]/width #ration 
    ratioheight=gray_scale_image.shape[0]/height #ration
    if ratiowidth > 1.5 and ratioheight > 1.5: #if image is big ---> rescale it
        dim = (width, height)    
        resized = cv2.resize(gray_scale_image, dim, interpolation = cv2.INTER_NEAREST)
        edge_gray_scale_image =cv2.Canny(resized,100,150)
        print(ratiowidth,ratioheight)    
    th2 = cv2.adaptiveThreshold(gray_scale_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)         #adabtive thresholding for binarization
    edge_gray_scale_image=cv2.medianBlur(edge_gray_scale_image,3) #median blur to remove noise
    cv2.imshow('img',edge_gray_scale_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
           
    kernalx=np.ones((1,6)) #this part mainly to keep H edges only
    imagex=cv2.morphologyEx(edge_gray_scale_image, cv2.MORPH_OPEN, kernalx)
    kernalx=np.ones((1,20))
    imagex=cv2.morphologyEx(imagex, cv2.MORPH_CLOSE, kernalx)
    
    
    cv2.imshow('img',imagex)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #this part mainly to keep V edges only
    kernaly=np.ones((7,1))
    imagey=cv2.morphologyEx(edge_gray_scale_image, cv2.MORPH_OPEN, kernaly)
    kernaly=np.ones((20,1))
    imagey=cv2.morphologyEx(imagey, cv2.MORPH_CLOSE, kernaly)
    
    
    cv2.imshow('img',imagey)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #collect them togather
    binaryimage = cv2.bitwise_or(imagey, imagex, mask = None) 
    
  
    cv2.imshow('img',binaryimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    #HOUGH lines to detect edges
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
    #if two lines are too close rho and theta --> avergage them
    for i in range(len(newlines)-1):
        if np.abs(newlines[i][0]-newlines[i+1][0])<15 and np.abs(newlines[i][1]-newlines[i+1][1])<.022:
            linerho=(newlines[i][0]+newlines[i+1][0])/2
            linetheta=(newlines[i][1]+newlines[i+1][1])/2
            newlines[i][0]=linerho
            newlines[i+1][0]=linerho
            newlines[i][1]=linetheta
            newlines[i+1][1]=linetheta
    
    
    binaryimage=cv2.bitwise_not(binaryimage)
    
    #draw this lines
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
    #find all contoures of the cleaned image
    contours=find_contours(binaryimage,.8)
    
    
    allcells=[]
    for cont in contours:
        xup=int(np.min(cont[:,0]))
        xdown=int(np.max(cont[:,0]))
        yleft=int(np.min(cont[:,1]))
        yright=int(np.max(cont[:,1]))
        width=yright-yleft
        height=xdown-xup
        if(width>20 and width<300 and height >10 ):
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

def extraxctidname(gray_scale_image):   
    edge_gray_scale_image =cv2.Canny(gray_scale_image,100,150)
    cv2.imshow('img',edge_gray_scale_image)
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
        if(height<3 or width<3):
            continue
        if width/height>6 and width/height <6.5:
            print(width/height)
            nameid.append([xup,xdown,yleft,yright])
    
    if nameid[0][2] < nameid[1][2]:
        return gray_scale_image[nameid[0][0]:nameid[0][1],nameid[0][2]:nameid[0][3]],gray_scale_image[nameid[1][0]:nameid[1][1],nameid[1][2]:nameid[1][3]]
    else:
        return gray_scale_image[nameid[1][0]:nameid[1][1],nameid[1][2]:nameid[1][3]],gray_scale_image[nameid[0][0]:nameid[0][1],nameid[0][2]:nameid[0][3]]
        
def mcqpreprocessing(path):
    colorimage= cv2.imread(path)#read the file
    gray_scale_image=cv2.cvtColor(colorimage,cv2.COLOR_BGR2GRAY)  #convert it to gray scale   
    width = 720 #perfect width to our application
    height = 960 #perfect height 
    ratiowidth=colorimage.shape[1]/width #ration 
    ratioheight=colorimage.shape[0]/height #ration    
    dim = (width, height)    
    if ratiowidth > 1.5 and ratioheight > 1.5: #if image is big ---> rescale it
        resized = cv2.resize(gray_scale_image, dim, interpolation = cv2.INTER_NEAREST)
        gray_scale_image=resized
        edge_gray_scale_image =cv2.Canny(resized,100,150)
    print(ratiowidth,ratioheight)          
    cv2.imshow('img',edge_gray_scale_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    cnts, hierarchy = cv2.findContours(edge_gray_scale_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]
    rects = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        if h >= 15:
            # if height is enough
            # create rectangle for bounding
            rect = (x, y, w, h)
            rects.append(rect)
            cv2.rectangle(resized, (x, y), (x+w, y+h), (0, 255, 0), 1);
            cv2.imshow('img',resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()             
            newimage= gray_scale_image[int(y):int((y+h)),int(x):int((x+w))]
            width = 720 #perfect width to our application
            height = 960 #perfect height 
            dim = (width, height)   
            resized = cv2.resize(newimage, dim, interpolation = cv2.INTER_NEAREST)
            return resized



def intersection(line1, line2):  
    rho1, theta1 = line1   
    rho2, theta2 = line2   
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]

 

def rotation(path,arg): 
    colorimage= cv2.imread(path)#read the file
    gray_scale_image=cv2.cvtColor(colorimage,cv2.COLOR_BGR2GRAY)  #convert it to gray scale
    edge_gray_scale_image =cv2.Canny(gray_scale_image,100,150) #canny edge detection
    width = 720 #perfect width to our application
    height = 960 #perfect height 
    ratiowidth=colorimage.shape[1]/width #ration 
    ratioheight=colorimage.shape[0]/height #ration
    if ratiowidth > 1.5 and ratioheight > 1.5: #if image is big ---> rescale it
        dim = (width, height)    
        resized = cv2.resize(gray_scale_image, dim, interpolation = cv2.INTER_NEAREST)
        edge_gray_scale_image =cv2.Canny(resized,100,150)
        print(ratiowidth,ratioheight)  
    cv2.imshow('img',edge_gray_scale_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cnts = cv2.findContours(edge_gray_scale_image.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    
    if arg==1:        
        extLeftup = tuple(c[c[:, :, 0].argmin()][0])
        extrightdown = tuple(c[c[:, :, 0].argmax()][0])
        extleftdown = tuple(c[c[:, :, 1].argmin()][0])
        extrightup = tuple(c[c[:, :, 1].argmax()][0])
    elif arg==2:
        extrightup = tuple(c[c[:, :, 0].argmin()][0])
        extleftdown = tuple(c[c[:, :, 0].argmax()][0])
        extrightdown = tuple(c[c[:, :, 1].argmin()][0])
        extLeftup = tuple(c[c[:, :, 1].argmax()][0])
    
      
    width=resized.shape[1]
    
    height=resized.shape[0]
    rect=np.array([
                extLeftup,extrightup,extrightdown,extleftdown
                ],dtype = "float32")
    dst = np.array([
    		[0, 0],
    		[width-1, 0],
    		[width-1, height - 1],
    		[0, height - 1]], dtype = "float32") 
    	# compute the perspective transform matrix and then apply it
    M=cv2.getPerspectiveTransform(rect, dst)
    deskewedimage = cv2.warpPerspective(resized, M, (width, height)) 
    flipHorizontal = cv2.flip(deskewedimage, 0)
    cv2.imshow('img',flipHorizontal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return flipHorizontal

