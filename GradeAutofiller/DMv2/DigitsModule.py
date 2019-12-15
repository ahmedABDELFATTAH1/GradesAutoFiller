import pickle
from commonfunctions import *
from skimage.feature import hog
from sklearn.preprocessing import normalize
import skimage
from skimage.morphology import binary_erosion, binary_dilation, binary_closing,skeletonize, thin,erosion,dilation
from skimage.transform import rotate
from skimage.measure import find_contours

model = pickle.load(open('hognn_model.sav', 'rb'))

def hog_features(X, imgshape=(28, 28), pixels_per_cell=(6, 6)):
    features = []
    for row in X:
        img = row.reshape(imgshape)
        img_feature = hog(img, orientations=8, pixels_per_cell=pixels_per_cell, cells_per_block=(2, 2))
        features.append(img_feature)
    return np.array(features)

def predict_img(img):
    img = rgb2gray(img)
    img = (img.reshape(1,-1))
    Xhog = hog_features(img)
    Xhog = normalize(Xhog)
    Y = (model.predict(Xhog))
    return (np.argmax(Y))

def digits_locv2(img,X=28,Y=28):
    """
    Localization of the digits , it separates each digit into a fixed size output
    Arugments :
    -- img : numpy array
    Returns
    -- digits :  Array of fixed size matrices for each digit .
    """
    img = rgb2gray(img) 

    X = int(X)
    Y = int(X)
    
    Xh = int(X/2)
    Yh = int(Y/2)
    if(img.shape[0]>28):
        img = skimage.transform.resize(img, (28, img.shape[1]))
        io.imshow(img)
        img = img != 0
    #img = thin(img)
    labeled_array = measure.label(img)
    n = np.amax(labeled_array)
    digits = {}
    h,w = img.shape
    for i in range (1 , n+1):
        
        digit = labeled_array == i
        white_pixels = np.array(np.where(digit == 1))
        Ymin,Xmin = white_pixels[:,0]
        Ymax,Xmax = white_pixels[:,-1]
        shaped_digit = np.zeros([X,Y])
        
        Sx = max(0,int(Ymin-3))
        Fx = min(Ymax+3,h)
        Dx = Fx-Sx
        
        Sy = max(0,int(Xmin-3))
        Fy = min(w,Xmax+3)
        Dy = Fy-Sy
        
        digit = (digit[Sx:Fx+1,Sy:Fy+1])
        if (digit.shape[1] > 28):
            digit = skimage.transform.resize(digit, (digit.shape[0],28))
        
        shaped_digit[ :digit.shape[0] ,:digit.shape[1]] = digit
            
        digits[Xmin]=shaped_digit
    output = []
    for i in sorted (digits):
        output.append(1 - digits[i])
    return output
    
def digits_locv1(img,X=28,Y=28):
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
    contours = find_contours(img, 0.7,fully_connected='high',positive_orientation='high')
    for n, contour in enumerate(contours):
        
        #print(len(contour))
        if(len(contour) < 50) :
            continue
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
    contours = find_contours(img, 0.7,fully_connected='high',positive_orientation='high')
    for n, contour in enumerate(contours):
        
       # print(len(contour))
        if(len(contour) < 40) :
            continue
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
    
def rec_cell(img):
    digit_strv1 = ""
    digit_strv2 = ""
    img = rgb2gray(img)
    NotThin = (np.amax(thin(img)-img))
    
    imgv1 = np.copy(img)
    imgv1 = erosion(imgv1)
    imgv1 = binary_dilation(imgv1)
    digitsv1 = digits_locv1(imgv1)
    #show_images(digitsv1)
    
    imgv2 = np.copy(img)
    imgv2 = rgb2gray(imgv2)
    imgv2 = erosion(imgv2)
    digitsv2 = digits_locv1(imgv2)
    #show_images(digitsv2)
    for dig in digitsv1:
        digit_strv1 += str(predict_img(dig))
    
    for dig in digitsv2:
        digit_strv2 += str(predict_img(dig))
    
    if(len(digit_strv1)!= "" and NotThin != 0):
        return digit_strv1
    else:
        return digit_strv2
    digit_strv1 = ""
    digit_strv2 = ""
    
    imgv1 = np.copy(img)
    imgv1 = erosion(imgv1)
    imgv1 = binary_dilation(imgv1)
    digitsv1 = digits_locv1(imgv1)
    
    imgv2 = np.copy(img)
    imgv2 = rgb2gray(imgv2)
    imgv2 = erosion(imgv2)
    digitsv2 = digits_locv1(imgv2)
    
    for dig in digitsv1:
        digit_strv1 += str(predict_img(dig))
    
    for dig in digitsv2:
        digit_strv2 += str(predict_img(dig))
    
    if(len(digit_strv1)!= ""):
        return digit_strv1
    else:
        return digit_strv2