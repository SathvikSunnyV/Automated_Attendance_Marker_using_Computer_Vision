import os
import cv2
from config import ALLOWED_EXTENSIONS

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Invalid image file")
    return image
