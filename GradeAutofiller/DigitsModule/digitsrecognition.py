import pickle
from commonfunctions import *
from skimage.feature import hog
from sklearn.preprocessing import normalize

model = pickle.load(open('hog_model.sav', 'rb'))

def get_number(img):
    img_features = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
    x_digit = np.array([img_features])
    x_digit = normalize(x_digit)
    y_digit = model.predict_classes(x_digit)
    return y_digit[0]
