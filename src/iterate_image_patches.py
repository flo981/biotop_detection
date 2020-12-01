#go through biotop_dir, show image,ask label
import cv2
import os
from shutil import copyfile
import numpy as np
import imutils

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])




def iterate_patches(files, bio_folder):
    borderfile = [x for x in files if "B" in x  and "cropped" not in x][0]
    rawfile = [x for x in files if "M" not in x  and "cropped" not in x if "B" not in x][0]
    cropfile = [x for x in files if "crop"  in x][0]
    print(borderfile,rawfile,cropfile)
    # Open image with red border line and make into Numpy array
    #im = Image.open('../data/output_biotop_dir/' + bio_folder + '/' + borderfile).convert('RGB')
    image = cv2.imread('../data/output_biotop_dir/' + bio_folder + '/' + borderfile)
    image_raw = cv2.imread('../data/output_biotop_dir/' + bio_folder + '/' + rawfile)
    image_mask = cv2.imread('../data/output_biotop_dir/' + bio_folder + '/' + cropfile,cv2.IMREAD_UNCHANGED)

    ##SLIDING WINDOW
    (winW, winH) = (32, 32)
    # loop over the image pyramid
    for resized in pyramid(image_mask, scale=100):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW
            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            #time.sleep(0.025)


            finish = False
            while not finish:
                cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                cv2.imshow("Window", clone)
                key = cv2.waitKey(0)
                if key == ord('1'):
                    print('1')
                    src = '../data/output_biotop_dir/' + bio_folder + '/' + cropfile
                    dst = '../data/training_dataset/hedge_100/' + 't_' + cropfile
                    cv2.imwrite("test.png",window)
                    #copyfile(src, dst)
                    finish = True
                elif key == ord('2'):
                    print('2')
                    src = '../data/output_biotop_dir/' + bio_folder + '/' + cropfile
                    dst = '../data/training_dataset/hedge_50/' + 't_' + cropfile
                    #copyfile(src, dst)
                    finish = True
                elif key == ord('3'):
                    print('3')
                    src = '../data/output_biotop_dir/' + bio_folder + '/' + cropfile
                    dst = '../data/training_dataset/hedge_0/' + 't_' + cropfile
                    #copyfile(src, dst)
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
            iterate_patches(files, folder)
            #crop_v2(files,folder)

#txt = input("Type something to test this out: ")
