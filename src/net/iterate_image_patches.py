# go through biotop_dir, show image,ask label
import cv2
import os
from shutil import copyfile
import numpy as np
import imutils

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import argparse
import time


import numpy as np
from PIL import Image
import tensorflow as tf  # TF2





def pyramid(image, scale=1.5, minSize=(20, 20)):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        if w==0:
            w=1
            print("Excepten w:0->1")
            break
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


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def iterate_patches(files, bio_folder, args, height, width):
    borderfile = [x for x in files if "B" in x and "cropped" not in x][0]
    rawfile = [x for x in files if "M" not in x and "cropped" not in x if "B" not in x][0]
    cropfile = [x for x in files if "C" in x][0]
    # Open image with red border line and make into Numpy array
    # im = Image.open('../data/output_biotop_dir/' + bio_folder + '/' + borderfile).convert('RGB')
    image = cv2.imread('../../data/output_biotop_dir/' + bio_folder + '/' + borderfile)
    image_raw = cv2.imread('../../data/output_biotop_dir/' + bio_folder + '/' + rawfile)
    image_mask = cv2.imread('../../data/output_biotop_dir/' + bio_folder + '/' + cropfile, cv2.IMREAD_UNCHANGED)
    write_count = 1
    c_bio = c_nbio = 0
    # SLIDING WINDOW
    (winW, winH) = (16, 16)
    labels = load_labels(args.label_file)
    # loop over the image pyramid
    for resized in pyramid(image_mask, scale=1000):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=winW, windowSize=(winW, winH)):
            mod_im = resized
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            clone = resized.copy()

            # Disregard windows outside of biotop/border cases
            threshold_black = 100
            gray_version = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
            if (cv2.countNonZero(gray_version) > threshold_black):
            #    cv2.imwrite("../data/_temp_sliding_window/"+str(write_count)+".png",window)
            #    write_count = write_count +1
            # img = Image.open(args.image)
                img = Image.fromarray(window).convert('RGB')
                img = img.resize([width, height])

                #START CLASSIFICATION PROCESS
                # add N dim
                input_data = np.expand_dims(img, axis=0)
                if floating_model:
                    input_data = (np.float32(input_data) - args.input_mean) / args.input_std

                interpreter.set_tensor(input_details[0]['index'], input_data)

                interpreter.invoke()

                output_data = interpreter.get_tensor(output_details[0]['index'])
                results = np.squeeze(output_data)

                top_k = results.argsort()[-5:][::-1]
                cv2.imshow("window", mod_im)
                key = cv2.waitKey(0)

                if top_k[0] == 1:
                    #no bio
                    c_nbio = c_nbio+1
                    mod_im = cv2.rectangle(mod_im, (x, y), (x + winW, y + winH), (0, 0, 255), 1)
                if top_k[0] == 0:
                    #bio
                    c_bio = c_bio+1
                    mod_im = cv2.rectangle(mod_im, (x, y), (x + winW, y + winH), (0, 255, 0), 1)
                # for i in top_k:
                #     if floating_model:
                #         print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
                #     else:
                #         print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

        percent = (c_bio*100)/(c_bio+c_nbio)
        content  = "{} {:02.2f} {}".format(bio_folder[4:], percent, c_bio+c_nbio)
        print(content)

        #show results in patch
        cv2.imshow("window", mod_im)
        key = cv2.waitKey(0)

        content  = "{} {:02.2f} {}".format(bio_folder[4:], percent, c_bio+c_nbio)
        return content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image',
        default='/tmp/grace_hopper.bmp',
        help='image to be classified')
    parser.add_argument(
        '-m',
        '--model_file',
        default='/tmp/mobilenet_v1_1.0_224_quant.tflite',
        help='.tflite model to be executed')
    parser.add_argument(
        '-l',
        '--label_file',
        default='/tmp/labels.txt',
        help='name of file containing labels')
    parser.add_argument(
        '--input_mean',
        default=127.5, type=float,
        help='input_mean')
    parser.add_argument(
        '--input_std',
        default=127.5, type=float,
        help='input standard deviation')
    parser.add_argument(
        '--num_threads', default=None, type=int, help='number of threads')
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(
        model_path=args.model_file, num_threads=args.num_threads)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    PATH = '../../data/output_biotop_dir/'
    folders = os.listdir(PATH)
    i = 0

    #write results to file
    output_file = open('output.txt', 'w')
    output_file.write("BiotopNR Percentage nPatches\n")

    # Remove ".DS_Store"
    folders[:] = [x for x in folders if ".DS_Store" not in x]
    for folder in folders:
        i = i + 1
        files = os.listdir(PATH + folder)
        files[:] = [x for x in files if ".DS_Store" not in x and ".txt" not in x]
        if (len(files) != 1 and len(files) != 0):
            content = iterate_patches(files, folder, args, height, width)
            output_file.write(content + "\n")
        #break
        if i == 1000:
            output_file.close()
            break

    output_file.close()
        # crop_v2(files,folder)
    # with open('output.txt', 'w') as file:
    #     file.write(json.dumps(output_dict)) # use `json.loads` to do the reverse


# txt = input("Type something to test this out: ")
