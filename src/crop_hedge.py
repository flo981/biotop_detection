
#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image, ImageFilter
import cv2
import matplotlib.pyplot as plt



#TODO: Maske auf orginalBild ohne rote linien anwenden
#Oder rote linien dünner?
#bessere qualität!
#wenn hecke komplexer => problem da nur die hülle ausgeschnitten wird!
#schwarze linien => folyfit confex hull check (wahrscheinlicher grund: lücken in roter linie)
#Größerer Rahmen ausscheniden: mit addieren! beachten ob links oder recht (oben/unten)
# von centrum, dementsprechend +oder- addieren zum wert
# ML-NN: Train on color and structure?

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
    # #F700FF = [247, 0, 255]
    # #FF0000 = [255,0,0]
    x_border, y_border = np.where(np.all(na==[255, 0, 0],axis=2))
    temp = np.array([y_border, x_border])
    temp = np.transpose(temp)
    # original image
    # -1 loads as-is so if it will be 3 or 4 channel as the original
    image = cv2.imread('../data/output_biotop/'+file, -1)


    #roi_corners = np.array([[(x_border[0],y_border[0]), (x_border[10],y_border[10]), (x_border[20],y_border[20]), (x_border[20],y_border[20])]], dtype=np.int32)
    roi_corners_new = np.array([[somestuff(k) for k in temp]], dtype=np.int32)
    roi_corners = roi_corners_new

    # mask defaulting to black for 3-channel and transparent for 4-channel
    mask = np.zeros(image.shape, dtype=np.uint8)
    #(1, 4, 2) =4 einträge ,x+y
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillConvexPoly(mask, cv2.convexHull(roi_corners), ignore_mask_color)

    cv2.imwrite('../data/output_biotop_crop/croppeddssd_'+file, mask)

    # from Masterfool: use cv2.fillConvexPoly if you know it's convex
    # fillPoly


    # apply the mask
    masked_image = cv2.bitwise_and(image, mask)

    xmin = min(x_border)
    ymin = min(y_border)
    xmax = max(x_border)
    ymax = max(y_border)

    masked_cropped_image = masked_image[xmin:xmax, ymin:ymax]

    # save the result
    cv2.imwrite('../data/output_biotop_crop/cropped_'+file, masked_cropped_image)

if __name__ == "__main__":
    files = os.listdir('../data/output_biotop/')
    for file in files:
        if file.endswith('.png'):
            print('crop: '+file)
            cropImage(file)

