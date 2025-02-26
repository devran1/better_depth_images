import matplotlib.pyplot as plt
import numpy as np
import cv2

#1.13.0+cpu
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings('ignore')

path="image.jpg"

image = cv2.imread(path) #generate the mask

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



zero=0
object_number=12

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


plt.imshow(mask, interpolation='none')
plt.imsave(f"{path}-mask.jpg",mask)
plt.show()

