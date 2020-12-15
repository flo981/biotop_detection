import cv2
import os


def mouse_click(event, x, y,
                flags, param):
    # to check if left mouse
    # button was clicked
    global finish, count
    if event == cv2.EVENT_LBUTTONDOWN:
        count = count +1
        window = 20
        patch = image_copy[y-window:y + window, x-window:x + window]
        cv2.imshow("patch", patch)

        #convert image to RGBA and make black transparent
        src = patch
        tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(src)
        rgba = [b, g, r, alpha]
        dst = cv2.merge(rgba, 4)

        if case == 1:
            print("Biotop at: ",x,y,' #Patch: ',count)
            filename = '../../data/training_patches_80_3_classes/bio/'+str(count)+'_bio.png'
            cv2.imwrite(filename, dst)

        if case == 2:
            print("No Biotop at: ",x,y,' #Patch: ',count)
            filename = '../../data/training_patches_80_3_classes/no_bio/'+str(count)+'_no_bio.png'
            cv2.imwrite(filename, dst)

        if case == 3:
            print("No Biotop at: ",x,y,' #Patch: ',count)
            filename = '../../data/training_patches_80_3_classes/no_bio/'+str(count)+'_no_bio.png'
            cv2.imwrite(filename, dst)


def crop_patch(files, bio_folder):
    cropfile = [x for x in files if "C"  in x][0]
    rawfile = [x for x in files if "C" not in x][0]

    image_mask = cv2.imread('../../data/output_biotop_dir/' + bio_folder + '/' + cropfile)
    image_raw = cv2.imread('../../data/output_biotop_dir/' + bio_folder + '/' + rawfile)

    global image_copy, finish, case
    image_copy = image_mask.copy()

    ####2_565282994
    #case = 1 : Hedge
    #case = 2 : no Hedge
    case = 0
    ####
    for i in range(1,3):
        case = i
        print("Case: ", case)
        print(rawfile)
        cv2.imshow('image', image_mask)
        image_raw = cv2.resize(image_raw, (720,480))  # Resize image
        cv2.imshow('im_raw',image_raw)
        cv2.setMouseCallback('image', mouse_click)
        finish = False
        while not finish:
            key = cv2.waitKey(0)

            if key == ord('1'):
                print('1')
                finish = True



if __name__ == "__main__":
    PATH = '../../data/output_biotop_dir/'
    folders = os.listdir(PATH)
    i=0
    count = 0
    #Remove ".DS_Store"
    folders[:] = [x for x in folders if ".DS_Store" not in x]
    for folder in folders:
        i=i+1
        files = os.listdir(PATH + folder)
        files[:] = [x for x in files if ".DS_Store" not in x and "B" not in x and "M" not in x and ".txt" not in x]
        #files[:] = [x for x in files if ".DS_Store" not in x and "cropped" not in x and "B" not in x]

        if (len(files) != 1 and len(files) != 0):
            print('crop: '+folder, i, "/", len(folders))
            crop_patch(files, folder)
            #crop_v2(files,folder)
