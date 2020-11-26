#go through biotop_dir, show image,ask label
import cv2
import os
from shutil import copyfile
import numpy as np

def label_file(files, bio_folder):
        borderfile = [x for x in files if "B" in x][0]
        rawfile = [x for x in files if "B" not in x][0]
        # Open image with red border line and make into Numpy array
        #im = Image.open('../data/output_biotop_dir/' + bio_folder + '/' + borderfile).convert('RGB')
        image = cv2.imread('../data/output_biotop_dir/' + bio_folder + '/' + borderfile)
        image_raw = cv2.imread('../data/output_biotop_dir/' + bio_folder + '/' + rawfile)

        image = cv2.resize(image, (720,480))  # Resize image
        image_raw = cv2.resize(image_raw, (720,480))  # Resize image

        finish = False
        while not finish:
            numpy_horizontal_concat = np.concatenate((image, image_raw), axis=1)
            cv2.imshow("result", numpy_horizontal_concat)

            key = cv2.waitKey(0)
            if key == ord('1'):
                print('1')
                src = '../data/output_biotop_dir/' + bio_folder + '/' + '2cropped_' + rawfile
                dst = '../data/training_dataset/hedge/' + 't_' + rawfile
                copyfile(src, dst)
            elif key == ord('2'):
                print('2')
                src = '../data/output_biotop_dir/' + bio_folder + '/' + '2cropped_' + rawfile
                dst = '../data/training_dataset/no_hedge/' + 't_' + rawfile
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
        files[:] = [x for x in files if ".DS_Store" not in x and "cropped" not in x]
        if (len(files) != 1 and len(files) != 0):
            print('crop: '+folder, i, "/", len(folders))
            label_file(files, folder)
            #crop_v2(files,folder)

#txt = input("Type something to test this out: ")
