from commonfunctions import *
from digitslocalization import *
from digitsrecognition import *

def rec_cell(img):
    output = ""
    digits = (digits_loc(img))
    for digit in digits :
        hist = histogram(digit, nbins=2)
        if ( hist[0][1]/(hist[0][1]+hist[0][0]) < .06 ):
            floatidx = output.find('.')
            if (floatidx == -1):
                output +="."
            continue 
        else :
            d = get_number(digit)
            output += str(d)
    return output