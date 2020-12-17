
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



def crop_masked(files,bio_folder):
    borderfile = [x for x in files if "M" in x][0]
    rawfile = [x for x in files if "M" not in x and ".txt" not in x and "C" not in x and "i" not in x][0]
    # Open image with red border line and make into Numpy array
    image_mask = cv2.imread('../../data/output_biotop_dir/' + bio_folder + '/' + borderfile)
    image_raw = cv2.imread('../../data/output_biotop_dir/' + bio_folder + '/' + rawfile)

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

    filename = '../../data/output_biotop_dir/' + bio_folder + '/' + 'C_' + rawfile
    # save the result

    cv2.imwrite(filename, dst)

if __name__ == "__main__":
    PATH = '../../data/output_biotop_dir/'
    folders = os.listdir(PATH)
    i=0
    #Remove ".DS_Store"
    folders[:] = [x for x in folders if ".DS_Store" not in x and ".txt" not in x]
    for folder in folders:
        i=i+1
        files = os.listdir(PATH + folder)
        files[:] = [x for x in files if ".DS_Store" not in x and "cropped" not in x and "B" not in x]
        if (len(files) != 1 and len(files) != 0):
            print('crop: '+folder, i, "/", len(folders))
            #cropImage(files, folder)
            crop_masked(files, folder)
