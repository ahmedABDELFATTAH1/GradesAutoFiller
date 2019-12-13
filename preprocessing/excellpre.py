import numpy as np
import cv2
import skimage.io as io
from skimage.color import rgb2gray
from numba import vectorize, cuda
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.exposure import histogram
from skimage.measure import find_contours
#from skimage.draw import rectangle, line
from skimage.transform import rotate
from skimage.filters import threshold_local,median
from skimage.transform import hough_line, hough_line_peaks

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


def preprocessing(path):
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
    th3 = cv2.adaptiveThreshold(img_rotation,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    xlines=[]
    ylines=[]
    lines=cv2.HoughLines(edge_gray_scale_image,1,np.pi/180,250)   
    for line in lines:    
        rho,theta = line[0]       
        angle=((theta*180)/np.pi)+180
        if (angle > 70 and angle < 120) or (angle> 240 and angle <300): 
            xlines.append([rho,theta])             
        elif (angle > 330) or(angle < 30) or (angle <210 and angle >150):
            ylines.append([rho,theta])
        filteredxlines=[]
        xlines=sorted(xlines,key=lambda l:l[0])        
        prev_rho_x =100000
        for line in xlines:
            rho=line[0]
            print(rho)
            if rho-prev_rho_x <10:
                prev_rho_x=rho
                continue
            else:
                filteredxlines.append(line)
                prev_rho_x=rho
    filteredylines=[]
    ylines=sorted(ylines,key=lambda l:l[0])        
    prev_rho_y =100000
    for line in ylines:
        rho=line[0]
        print(rho)
        if np.abs(prev_rho_y-rho) <12:
            prev_rho_y=rho
            continue
        else:
            filteredylines.append(line)
            prev_rho_y=rho
    rowscells = []
    for i in range(len(filteredxlines)-1):
        rowscells.append([])
        linexup=filteredxlines[i]  
        linexdown=filteredxlines[i+1]    
        for j in range(1,len(filteredylines)):
            lineyleft=filteredylines[j]
            lineyright=filteredylines[j-1]
            pointupleft=intersection(linexup,lineyleft)
            pointdownleft=intersection(linexdown,lineyleft)
            pointupright=intersection(linexup,lineyright)
            pointdownright=intersection(linexdown,lineyright)         
            width=int(np.sqrt(np.power(pointdownright[0]-pointdownleft[0],2)+np.power(pointdownright[1]-pointdownleft[1],2)))
            height=int(np.sqrt(np.power(pointupleft[0]-pointdownleft[0],2)+np.power(pointupleft[1]-pointdownleft[1],2)))  
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
            ped = cv2.warpPerspective(th3, M, (width, height))     
            rowscells[i].append(ped)
    return rowscells


if __name__ =='__main__':
    rowcells=preprocessing('excelpic/8.jpg')
    
    
    

