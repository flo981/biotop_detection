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


def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping

    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False  # cropping is finished

        refPoint = [(x_start, y_start), (x_end, y_end)]

        if len(refPoint) == 2:  # when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", roi)

def select_patches(files, bio_folder):
    borderfile = [x for x in files if "B" in x  and "cropped" not in x][0]
    rawfile = [x for x in files if "M" not in x  and "cropped" not in x if "B" not in x][0]
    cropfile = [x for x in files if "crop"  in x][0]
    print(borderfile,rawfile,cropfile)
    # Open image with red border line and make into Numpy array
    #im = Image.open('../data/output_biotop_dir/' + bio_folder + '/' + borderfile).convert('RGB')
    image = cv2.imread('../data/output_biotop_dir/' + bio_folder + '/' + borderfile)
    image_raw = cv2.imread('../data/output_biotop_dir/' + bio_folder + '/' + rawfile)
    image_mask = cv2.imread('../data/output_biotop_dir/' + bio_folder + '/' + cropfile)


    cropping = False
    x_start, y_start, x_end, y_end = 0, 0, 0, 0

    oriImage = image_mask.copy()



    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)

    while True:

        i = image_mask.copy()

        if not cropping:
            cv2.imshow("image", image_mask)

        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", i)

        cv2.waitKey(1)

    # close all open windows
    cv2.destroyAllWindows()



if __name__ == "__main__":
    PATH = '../data/output_biotop_dir/'
    folders = os.listdir(PATH)
    i=0
    global x_start, y_start, x_end, y_end, cropping


    #Remove ".DS_Store"
    folders[:] = [x for x in folders if ".DS_Store" not in x]
    for folder in folders:
        i=i+1
        files = os.listdir(PATH + folder)
        files[:] = [x for x in files if ".DS_Store" not in x]
        if (len(files) != 1 and len(files) != 0):
            print('crop: '+folder, i, "/", len(folders))
            select_patches(files, folder)
            #crop_v2(files,folder)

