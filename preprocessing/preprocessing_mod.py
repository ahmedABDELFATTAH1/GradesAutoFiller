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
from excellpre import preprocessing,returncell
from skimage.morphology import skeletonize

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

def deskewImage(path):    
    colorimage= cv2.imread(path)
    gray_scale_image=cv2.cvtColor(colorimage,cv2.COLOR_BGR2GRAY)
    edge_gray_scale_image =cv2.Canny(gray_scale_image,100,150)
    H, theta, d = hough_line(edge_gray_scale_image)
    _,angles,distance=hough_line_peaks(H,theta,d,num_peaks=1)
    angle=int(angles[0]*180/np.pi)
    rotation_matrix = cv2.getRotationMatrix2D((gray_scale_image.shape[1]/2,gray_scale_image.shape[0]/2), angle, 1)
    img_rotation = cv2.warpAffine(gray_scale_image, rotation_matrix, (gray_scale_image.shape[1], gray_scale_image.shape[0]))
    colorimage=cv2.warpAffine(colorimage, rotation_matrix, (gray_scale_image.shape[1], gray_scale_image.shape[0]))
    edge_gray_scale_image =cv2.Canny(img_rotation,100,150)
    
    
    cv2.imshow('img',edge_gray_scale_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    xlines=[]
    ylines=[]
    lines=cv2.HoughLines(edge_gray_scale_image,1,np.pi/180,200) 
    
    for line in lines:    
        rho,theta = line[0]  
        angle=((theta*180)/np.pi)+180
        if (angle > 80 and angle < 100) or (angle> 260 and angle <280): 
            xlines.append([rho,theta])             
        elif (angle > 320) or(angle < 20) or (angle <200 and angle >160):
            ylines.append([rho,theta])      
            
    
    xlines=sorted(xlines,key=lambda l:l[0])  
    threshodlinex=5
    while(1):        
        filteredxlines=[]     
        prev_rho_x =100000
        print(len(filteredxlines))
        for line in xlines:
            rho=line[0]        
            if rho-prev_rho_x <threshodlinex:
                prev_rho_x=rho
                continue
            else:
                filteredxlines.append(line)
                prev_rho_x=rho
        print(len(filteredxlines))            
        if len(filteredxlines)>= 35 and  len(filteredxlines)<=42:
            break
        elif len(filteredxlines)<35:
            threshodlinex-=1
        else:
            threshodlinex+=1
            
    ylines=sorted(ylines,key=lambda l:l[0])  
    threshodliney=20      
    filteredylines=[]     
    prev_rho_y =100000
    for line in ylines:
        rho=line[0]   
        if np.abs(prev_rho_y-rho) <threshodliney:
            prev_rho_y=rho
            continue
        else:
            filteredylines.append(line)
            prev_rho_y=rho
            
    leftline=[10000,30]
    rightline=filteredylines[1]
    for line in filteredylines:
        if(line[0]>0 and line[0]<leftline[0]):
            leftline=line
    
    upline=filteredxlines[0]
    downline=filteredxlines[len(filteredxlines)-1] 
    pointupleft=intersection(upline,leftline)
    pointdownleft=intersection(downline,leftline)
    pointupright=intersection(upline,rightline)
    pointdownright=intersection(downline,rightline)  
    print(pointupleft)       
    width=img_rotation.shape[1]
    print(width)
    height=img_rotation.shape[0]
    rect=np.array([
                pointupleft,pointupright,pointdownright,pointdownleft
                ],dtype = "float32")
    dst = np.array([
    		[0, 0],
    		[width-1, 0],
    		[width-1, height - 1],
    		[0, height - 1]], dtype = "float32") 
    	# compute the perspective transform matrix and then apply it
    M=cv2.getPerspectiveTransform(rect, dst)
    deskewedimage = cv2.warpPerspective(img_rotation, M, (width, height)) 
    cv2.imshow('img',deskewedimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return deskewedimage 
 
if __name__=='__main__':
    grayimage=deskewImage('excelpic/1.jpg') 
    X_lines,Y_lines=preprocessing(grayimage)
    cell=returncell(5,8,X_lines,Y_lines,grayimage)
    cv2.imshow('img',cell)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    

    
    
    
    
    
    