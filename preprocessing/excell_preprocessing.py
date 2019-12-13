from libraries_mod import *

def linepeaks(image):
    H, theta, d = hough_line(image)
    _,angles,distance=hough_line_peaks(H,theta,d)
    return angles,distance
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

def edgesfilter(image):
    edgefilterx = np.array([[1, 1,2,1,1], [0,0,0,0,0], [-1,-1,-2,-1,-1]])
    edgefiltery = np.array([[1,0,-1], [1,0,-1], [2,0,-2],[1,0,-1], [1,0,-1]])
    edgesimagex=np.abs(sobelFilter(image,edgefilterx))
    edgesimagey=np.abs(sobelFilter(image,edgefiltery))
    edgesimagex=edgesimagex>300
    edgesimagey=edgesimagey>300
    openingfiltery=np.ones((20,1))
    openingfilterx=np.ones((1,20))
    edgesimagey=opening = cv2.morphologyEx(edgesimagey, cv2.MORPH_OPEN, openingfiltery)
    edgesimagex=opening = cv2.morphologyEx(edgesimagex, cv2.MORPH_OPEN, openingfilterx)
    return edgesimagex,edgesimagey

def rotatingexcellsheet(path):
    colorimage = io.imread(path)
    grayimage=rgb2gray(colorimage)
    grayimage=grayimage*255
    grayimage=grayimage.astype('uint8')
    edgedimage =cv2.Canny(grayimage,150,200)
    edgedimage=edgedimage==255
    angle = hughspace(edgedimage, 1)
    rotatedimg=rotate(edgedimage,90-angle)
    colorimage=rotate(colorimage,90-angle)
    return colorimage

