import numpy as np
import skimage.io as io
from skimage.color import rgb2gray
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
        thr1=helpfunc(H,0,thr)
        thr2=helpfunc(H,thr,256)        
        newthr=int((thr1.astype(int)+thr2.astype(int))/2)
        if newthr == thr :
            break
        else :
            thr=newthr 
    print(thr)
    
    
image=io.imread('cufe.png')
grayimage=rgb2gray(image)
grayimage=grayimage*255
grayimage=grayimage.astype('uint8')
getthreshold(grayimage)