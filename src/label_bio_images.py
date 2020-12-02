#go through biotop_dir, show image,ask label
import cv2
import os
from shutil import copyfile
import numpy as np
import imutils


def label_file(files, bio_folder):
    borderfile = [x for x in files if "B" in x  and "cropped" not in x][0]
    rawfile = [x for x in files if "M" not in x  and "cropped" not in x if "B" not in x][0]
    cropfile = [x for x in files if "crop"  in x][0]
    print(borderfile,rawfile,cropfile)
    # Open image with red border line and make into Numpy array
    #im = Image.open('../data/output_biotop_dir/' + bio_folder + '/' + borderfile).convert('RGB')
    image = cv2.imread('../data/output_biotop_dir/' + bio_folder + '/' + borderfile)
    image_raw = cv2.imread('../data/output_biotop_dir/' + bio_folder + '/' + rawfile)
    image_mask = cv2.imread('../data/output_biotop_dir/' + bio_folder + '/' + cropfile)


    image = cv2.resize(image, (720,480))  # Resize image
    image_raw = cv2.resize(image_raw, (720,480))  # Resize image

    finish = False
    while not finish:
        numpy_horizontal_concat = np.concatenate((image, image_raw), axis=1)
        #numpy_horizontal_concat = image_mask
        cv2.imshow("result", numpy_horizontal_concat)

        key = cv2.waitKey(0)
        if key == ord('1'):
            print('1')
            src = '../data/output_biotop_dir/' + bio_folder + '/' + cropfile
            dst = '../data/training_dataset/hedge_100/' + 't_' + cropfile
            copyfile(src, dst)
            finish = True
        elif key == ord('2'):
            print('2')
            src = '../data/output_biotop_dir/' + bio_folder + '/' + cropfile
            dst = '../data/training_dataset/hedge_50/' + 't_' + cropfile
            copyfile(src, dst)
            finish = True
        elif key == ord('3'):
            print('3')
            src = '../data/output_biotop_dir/' + bio_folder + '/' + cropfile
            dst = '../data/training_dataset/hedge_0/' + 't_' + cropfile
            copyfile(src, dst)
            finish = True


if __name__ == "__main__":
    PATH = '../data/output_biotop_dir/'
    folders = os.listdir(PATH)
    i=0

    #Remove ".DS_Store"
    folders[:] = [x for x in folders if ".DS_Store" not in x]
    for folder in folders:
        i=i+1
        files = os.listdir(PATH + folder)
        files[:] = [x for x in files if ".DS_Store" not in x]
        if (len(files) != 1 and len(files) != 0):
            print('crop: '+folder, i, "/", len(folders))
            label_file(files, folder)
            #crop_v2(files,folder)
