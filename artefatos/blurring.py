from skimage.util import random_noise
import numpy as np
import cv2

def to_0255(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
    return img.astype(np.uint8)

#Noise Levels
interval= 500
sigmasX=np.linspace(1.5,interval,10)
sigmasY=np.linspace(1.5,interval,10)
sigmasX=sigmasX/100
sigmasY=sigmasY/100

# deg_level de 1 a 10
def generate_blurring(img, deg_level):
    return cv2.GaussianBlur(img,(0,0), sigmaX=sigmasX[deg_level-1], sigmaY=sigmasY[deg_level-1], borderType=cv2.BORDER_CONSTANT)