import argparse
import os

import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

from net_helper import pyramid, sliding_window, write_csv


def classify_patch(image):
##################################################################
## image_classified, percent, nWindows = classify_patch(image)
## Input: -image (cv2.IMREAD_UNCHANGED format)
## Out:   -image_classified (cv2 image with labeled patches)
##        -percent (Percentage of positive biotop regions dedected)
##        -nWindows (Numburs of patches/sliding windows)
##################################################################

    (winW, winH) = (12, 12)
    c_bio = c_nbio = 0
    for resized in pyramid(image, scale=1000):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=winW, windowSize=(winW, winH)):
            mod_im = resized
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # Disregard windows outside of biotop/border cases
            threshold_black = 100
            gray_version = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
            if (cv2.countNonZero(gray_version) > threshold_black):

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


                if top_k[0] == 1:
                    #no bio
                    c_nbio = c_nbio+1
                    mod_im = cv2.rectangle(mod_im, (x, y), (x + winW, y + winH), (0, 0, 255), 1)
                if top_k[0] == 0:
                    #bio
                    c_bio = c_bio+1
                    mod_im = cv2.rectangle(mod_im, (x, y), (x + winW, y + winH), (0, 255, 0), 1)

        percent = (c_bio*100)/(c_bio+c_nbio)
        image_classified = cv2.cvtColor(mod_im, cv2.COLOR_RGBA2RGB)
        return image_classified, round(percent,2), c_bio+c_nbio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input_folder',
        default='/Users/flo/Proj/sat_hedge_detection/data/output_biotop_dir/',
        help='biotop current_biotop')
    parser.add_argument(
        '-m',
        '--model_file',
        default='/Users/flo/Proj/sat_hedge_detection/src/net/models/inception_v3_new_dataset_tuning.tflite',
        help='.tflite model to be executed')
    parser.add_argument(
        '-l',
        '--label_file',
        default='/Users/flo/Proj/sat_hedge_detection/src/net/models/labels.txt',
        help='name of file containing labels')
    parser.add_argument(
        '--input_mean',
        default=0, type=float,
        help='input_mean')
    parser.add_argument(
        '--input_std',
        default=255, type=float,
        help='input standard deviation')
    parser.add_argument(
        '--num_threads', default=None, type=int, help='number of threads')
    parser.add_argument(
        '--show_result', default=False, type=bool, help='show result im classification')
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


    biotop_counter = 0
    output_dict = {}

    # Prepare current_biotop_content
    biotop_names = os.listdir(args.input_folder)
    biotop_names[:] = [x for x in biotop_names if ".DS_Store" not in x]

    for current_biotop in biotop_names:
        ###################################################################################
        # current_biotop = eg bio_565272008
        # current_biotop_name = eg C_2_565272008.png
        # current_biotop_path /Users/flo/Proj/sat_hedge_detection/data/output_biotop_dir/
        #                       bio_565272008C_2_565272008.png
        ###################################################################################
        print("Process: ", current_biotop, "(", biotop_counter, "/",len(biotop_names),")")

        current_biotop_content = os.listdir(args.input_folder + current_biotop)
        # only select cropped biotope file (marked with a "C" eg: C_1_565275064.png)
        current_biotop_name = [x for x in current_biotop_content if "C" in x][0]
        current_biotop_path = args.input_folder + current_biotop + "/" + current_biotop_name

        image = cv2.imread(current_biotop_path, cv2.IMREAD_UNCHANGED)
        image_classified, percent, nWindows = classify_patch(image)
        print("Results: percent: ", percent, " nWindows", nWindows)

        if args.show_result:
            cv2.imshow("classified image", image_classified)
            cv2.waitKey(0)

        ## save results
        target_path = args.input_folder + current_biotop + "/" + "A_12_"+ current_biotop[4:] + ".png"
        image_classified = cv2.cvtColor(image_classified, cv2.COLOR_RGBA2RGB)
        cv2.imwrite(target_path,image_classified)

        output_dict[current_biotop] = {'percentage': percent, 'nPatches': nWindows}
        biotop_counter = biotop_counter + 1

    write_csv(output_dict)
    print("Done")
