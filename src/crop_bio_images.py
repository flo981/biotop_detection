
#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image, ImageFilter
import cv2
import matplotlib.pyplot as plt
import os.path

#vor cropv2
import numpy as np
from matplotlib.path import Path

# ML-NN: Train on color and structure?

def somestuff(k):
    return k

def cropImage(files, bio_folder):
    borderfile = [x for x in files if "B" in x][0]
    rawfile = [x for x in files if "B" not in x][0]
    # Open image with red border line and make into Numpy array
    im = Image.open('../data/output_biotop_dir/' + bio_folder + '/' + borderfile).convert('RGB')
    im = np.array(im)

    # Find X,Y coordinates of all red pixels
    # #F700FF = [247, 0, 255]
    # #FF0000 = [255,0,0]
    x_border, y_border = np.where(np.all(im==[255, 0, 0],axis=2))
    #plt.scatter(y_border, -x_border)
    #plt.show()


    temp = np.array([y_border, x_border])
    temp = np.transpose(temp)

    #roi_corners = np.array([[(x_border[0],y_border[0]), (x_border[10],y_border[10]), (x_border[20],y_border[20]), (x_border[20],y_border[20])]], dtype=np.int32)
    roi_corners_new = np.array([[somestuff(k) for k in temp]], dtype=np.int32)
    roi_corners = roi_corners_new
    #    #x = roi_corners[0][:, i]
    #plt.scatter(roi_corners[0][:, 0],roi_corners[0][:, 1])
    #plt.show()

    # load original image with cv2 (without red border line
    # -1 loads as-is so if it will be 3 or 4 channel as the original
    image = cv2.imread('../data/output_biotop_dir/' + bio_folder + '/' + rawfile, -1)

    # mask defaulting to black for 3-channel and transparent for 4-channel
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask_2 = np.zeros(image.shape, dtype=np.uint8)


    #(1, 4, 2) =4 einträge ,x+y
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    ignore_mask_color2 = (255,) * channel_count


    # fillConvexPoly if you know it's convex
    # fillPoly
    #cv2.fillConvexPoly(mask, cv2.convexHull(roi_corners), ignore_mask_color)
    cv2.fillConvexPoly(mask, cv2.convexHull(roi_corners), ignore_mask_color,lineType=cv2.LINE_AA)
    cv2.fillPoly(mask_2, roi_corners, ignore_mask_color2,lineType=cv2.LINE_AA)


    # apply the mask
    masked_image = cv2.bitwise_and(image, mask)
    masked_image_2 = cv2.bitwise_and(image, mask_2)

    #get edges to crop irrelevant regions
    xmin = min(x_border)
    ymin = min(y_border)
    xmax = max(x_border)
    ymax = max(y_border)

    masked_cropped_image = masked_image[xmin:xmax, ymin:ymax]
    masked_cropped_image_2 = masked_image_2[xmin:xmax, ymin:ymax]

    filename = '../data/output_biotop_dir/'+bio_folder + '/' +'1cropped_' + rawfile
    filename2 = '../data/output_biotop_dir/'+bio_folder + '/' +'2cropped_' + rawfile

    #try:
    print(" ==== > Crop: ", rawfile)
    # save the result
    cv2.imwrite(filename, masked_cropped_image)
    cv2.imwrite(filename2, masked_cropped_image_2)

    #except:
    #    print("error: ", rawfile)


def crop_masked(files,bio_folder):
    borderfile = [x for x in files if "M" in x][0]
    rawfile = [x for x in files if "M" not in x][0]

    # Open image with red border line and make into Numpy array
    image_mask = cv2.imread('../data/output_biotop_dir/' + bio_folder + '/' + borderfile)
    image_raw = cv2.imread('../data/output_biotop_dir/' + bio_folder + '/' + rawfile)

    #remove outliers
    image_mask = cv2.medianBlur(image_mask, 3)

    # Make all not white pixels black
    image_mask[np.all(image_mask != (255, 255, 255), axis=-1)] = (0,0,0)

    #apply mask
    masked_image = cv2.bitwise_and(image_raw, image_mask)

    # get edges to crop irrelevant regions
    border = np.where(np.all(image_mask == (255, 255, 255), axis=-1))

    xmin = min(border[0])
    ymin = min(border[1])
    xmax = max(border[0])
    ymax = max(border[1])

    masked_cropped_image = masked_image[xmin:xmax, ymin:ymax]

    #convert image to RGBA and make black transparent
    src = masked_cropped_image
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)

    filename = '../data/output_biotop_dir/' + bio_folder + '/' + '1cropped_' + rawfile
    print(" ==== > Crop: ", rawfile)
    # save the result

    cv2.imwrite(filename, dst)

if __name__ == "__main__":
    PATH = '../data/output_biotop_dir/'
    folders = os.listdir(PATH)
    i=0
    #Remove ".DS_Store"
    folders[:] = [x for x in folders if ".DS_Store" not in x]
    for folder in folders:
        i=i+1
        files = os.listdir(PATH + folder)
        files[:] = [x for x in files if ".DS_Store" not in x and "cropped" not in x and "B" not in x]
        if (len(files) != 1 and len(files) != 0):
            print('crop: '+folder, i, "/", len(folders))
            #cropImage(files, folder)
            crop_masked(files, folder)
