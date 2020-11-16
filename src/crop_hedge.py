
#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image, ImageFilter
import cv2


#TODO: Maske auf orginalBild ohne rote linien anwenden
#Oder rote linien dünner?
#bessere qualität!
#wenn hecke komplexer => problem da nur die hülle ausgeschnitten wird!
#schwarze linien => folyfit confex hull check (wahrscheinlicher grund: lücken in roter linie)


def somestuff(k):
    return k

def cropImage(file):
    # Open image and make into Numpy array
    im = Image.open('../data/output_biotop/'+file).convert('RGB')
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
    image = cv2.imread('../data/output_biotop/'+file, -1)

    # mask defaulting to black for 3-channel and transparent for 4-channel
    # (of course replace corners with yours)
    mask = np.zeros(image.shape, dtype=np.uint8)
    #roi_corners = np.array([[(yellowX[0],yellowY[0]), (yellowX[10],yellowY[10]), (yellowX[20],yellowY[20]), (yellowX[20],yellowY[20])]], dtype=np.int32)
    roi_corners_new = np.array([[somestuff(k) for k in temp]], dtype=np.int32)



    roi_corners = roi_corners_new
    #(1, 4, 2) =4 einträge ,x+y
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex

    # apply the mask
    masked_image = cv2.bitwise_and(image, mask)

    # save the result
    cv2.imwrite('../data/output_biotop_crop/croped_'+file, masked_image)

if __name__ == "__main__":
    files = os.listdir('../data/output_biotop/')
    for file in files:
        if file.endswith('.png'):
            print('crop: '+file)
            cropImage(file)
