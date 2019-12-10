import pytesseract 
from commonfunctions import *
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def en_name(image):
    """
    This function will handle the core OCR processing of getting english name.
    """
    text = pytesseract.image_to_string(image,lang='eng')  
    return text

def ar_name(image):
    """
    This function will handle the core OCR processing of getting arabic name.
    """
    text = pytesseract.image_to_string(image,lang='ara')  
    return text

def id_number(image):
    """
    This function will handle the core OCR processing of getting id number.
    """
    text = pytesseract.image_to_string(image,config='digits')  
    return text
