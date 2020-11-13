
#!/usr/bin/env python3

import numpy as np
from PIL import Image, ImageFilter
import cv2


#TODO: Maske auf orginalBild ohne rote linien anwenden
#Oder rote linien dünner?
#bessere qualität!


def somestuff(k):
    return k


# Open image and make into Numpy array
im = Image.open('../data/output_biotop/1_564050102.png').convert('RGB')
na = np.array(im)
orig = na.copy()    # Save original

# Median filter to remove outliers
im = im.filter(ImageFilter.MedianFilter(3))

# Find X,Y coordinates of all yellow pixels
yellowX, yellowY = np.where(np.all(na==[255,0,0],axis=2))
temp = np.array([yellowY, yellowX])
temp = np.transpose(temp)
# original image
# -1 loads as-is so if it will be 3 or 4 channel as the original
image = cv2.imread('../data/output_biotop/1_564050102.png', -1)
# mask defaulting to black for 3-channel and transparent for 4-channel
# (of course replace corners with yours)
mask = np.zeros(image.shape, dtype=np.uint8)
roi_corners = np.array([[(yellowX[0],yellowY[0]), (yellowX[10],yellowY[10]), (yellowX[20],yellowY[20]), (yellowX[20],yellowY[20])]], dtype=np.int32)
roi_corners_new = np.array([[somestuff(k) for k in temp]], dtype=np.int32)



#roi_corners_new_new = np.reshape(roi_corners_new, (1, 353, 2))

print(roi_corners.shape)
print(roi_corners_new.shape)

roi_corners = roi_corners_new
#(1, 4, 2) =4 einträge ,x+y
# fill the ROI so it doesn't get wiped out when the mask is applied
channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
ignore_mask_color = (255,)*channel_count
cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
# from Masterfool: use cv2.fillConvexPoly if you know it's convex

# apply the mask
masked_image = cv2.bitwise_and(image, mask)

# save the result
cv2.imwrite('image_masked.png', masked_image)
