from multiprocessing import current_process
from turtle import color, width
from xxlimited import foo
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle #save q-table
from matplotlib import style
import matplotlib.pyplot as plt

import time #for dynamic q-table
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
#print(os.listdir("./input"))

import pygame
import random
import time
import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings('ignore')

#1.13.0+cpu
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from PIL import Image
import torchvision.transforms as transforms
import torch

import math
import seaborn as sns
sns.set(rc={'figure.figsize' : (22, 10)})
sns.set_style("darkgrid", {'axes.grid' : True})

#https://stackoverflow.com/questions/14947909/python-checking-to-which-bin-a-value-belongs


#generated mask from original image, saved its colored version, get the depth map again
#then used it to generate another mask again...
#image = cv2.imread("58I3026FIH.jpg--", cv2.IMREAD_GRAYSCALE)
#image = cv2.imread("mask.jpg-edited-reversed.jpg", cv2.IMREAD_GRAYSCALE)

#image = cv2.imread("depth_image_0.png", cv2.IMREAD_GRAYSCALE)
#image = cv2.imread("58I3026FIH.jpg---edited-reversed.jpg", cv2.IMREAD_GRAYSCALE)
"""
#original
#path="KR8KS7KT7O.jpg--" #mask is being generated as KR8KS7KT7O.jpg---mask.jpg then depth generated
path="KR8KS7KT7O.jpg---mask.jpg-edited-reversed.jpg" #mask will be generated again..., but this time not from the original, this time from grayscale of the depth itself.
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
"""
#path="KR8KS7KT7O.jpg--" #<<<<path for generating better depth map to generate better, mask, we will use original depth map, for datasets, 
                            # simulation already has its own depth...
#image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

path="KR8KS7KT7O.jpg---mask-hist.jpg-edited-reversed.jpg"

image = cv2.imread(path) #generate the mask


"""
from PIL import Image, ImageEnhance

def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

enhanced_contrast = adjust_contrast(image, 1.5)

original_and_enhanced_contrast = np.hstack((image, enhanced_contrast))
image=original_and_enhanced_contrast
"""
"""

#https://www.section.io/engineering-education/image-adjustment-to-higher-or-lower-resolution-using-python/
img=image
layer = img.copy()
gp = [layer]
for j in range(2):
    layer = cv2.pyrDown(layer)
    gp.append(layer)

layer = gp[2]
#cv2.imshow("Gausian Upper level", layer)
lp = [layer]#Introduce and create a list for the Laplacian pyramid.

for j in range(2, 0, -1):
    Gausian_extended = cv2.pyrUp(gp[j])#Creating the Laplacian pyramid.
laplacian = cv2.subtract(gp[j-1], Gausian_extended)
#cv2.imshow(str(j), laplacian)


image=img
"""
#https://www.makeuseof.com/opencv-image-enhancement-techniques/


denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
contrast_stretched_image = cv2.normalize(denoised_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

# Image Sharpening
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
sharpened_image = cv2.filter2D(contrast_stretched_image, -1, kernel=kernel)

brightness_image = cv2.convertScaleAbs(sharpened_image, alpha=1, beta=5)

# Gamma Correction
gamma = 1.5
lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
gamma_corrected_image = cv2.LUT(brightness_image, lookup_table)

cv2.imwrite('final_image.jpg', gamma_corrected_image)


image = cv2.imread("final_image.jpg", cv2.IMREAD_GRAYSCALE)


#----------------
#histogram equalize


hist1 = cv2.calcHist([image],[0],None,[256],[0,256])
image = cv2.equalizeHist(image)

#-----------------------
#adaptive histogram equalize

"""

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
equalized = clahe.apply(image)
image=equalized
"""












"""
#cv2.imshow("Original image", img)
cv2.imwrite("hq-image.jpg",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


"""
# Gaussian Blur 
Gaussian = cv2.GaussianBlur(image, (10, 10), 0) 
#cv2.imshow('Gaussian Blurring', Gaussian) 
#cv2.waitKey(0) 
image=Gaussian
"""

"""

# Median Blur 
median = cv2.medianBlur(image, 5) 
image=median
"""

"""
bilateral = cv2.bilateralFilter(image, 10, object_number, object_number)   #kernel size, noise,noise?
image=bilateral
"""

#cv2.bilateralFilter(img, d, sigma_colour, sigma_space)
#img - The image that is loaded
#d - Diameter of the pixel neighborhood to consider while blurring. Since, this filter is slower than other filters,
# it is recommended to keep d = 5 for real-time applications.

#sigma_colour - This is the value of sigma in the colour space (RGB, HSV, and so on).
#Higher the value of this parameter, colours farther apart in the colour space are 
# considered for the filtering provided they lie within the range of sigma_space.                                                              

#sigma_space - This is the value of sigma in the co-ordinate space.
#  Higher this value, pixels farther apart in the co-ordinate space are considered 
# for the filtering provided their colours are within the sigma_colour range.


#mask of the depth (not producing good for the dataset) is the opinion, but we can work with it, if it doesn't then we can 
#mask the original image, then generate depth, then mask it again...

#depth of dataset
#image = cv2.imread("KR8KS7KT7O.jpg---edited-reversed.jpg", cv2.IMREAD_GRAYSCALE)



zero=0
object_number=12 #8 #+zero

mask=np.zeros((image.shape[0],image.shape[1]))

matrix_image=np.array(image)

uniques=np.unique(matrix_image)

threshold=20

bins=np.histogram(uniques,object_number)#+zero)

listis=uniques
binsare=bins[1]
place=np.digitize(listis,binsare)
print(place)


for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        place=np.digitize(image[i, j],binsare)
        mask[i][j]=place-1

#print(mask)

"""
for depth2.jpg
place=np.digitize(image[j, i],binsare)
IndexError: index 1024 is out of bounds for axis 0 with size 1024
"""

#try both of the images which one is better

"""
#PROOF IT WORKS
im=Image.fromarray(mask) #.show()
im.show()
#im=im.convert('RGB') #bad
#im.save(f"{path}-mask2.jpg") #bad
"""


plt.imshow(mask, interpolation='none')
plt.imsave(f"{path}-mask.jpg",mask)
plt.show()



"""
#plt.imshow(mask, interpolation='nearest') #interpolation doesn't matter

'antialiased', 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 
'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman'
"""
