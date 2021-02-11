import os
from shutil import copyfile
import csv
import numpy as np
from pyproj import Transformer


#inProj = Proj(init='epsg:4326')
#outProj = Proj(init='epsg:31258')

transformer = Transformer.from_crs("epsg:4326", "epsg:31258")

source_directory = '/Users/flo/Proj/sat_hedge_detection/data/output_biotop_dir'
target_directory = '/Users/flo/Proj/sat_hedge_detection/data/output_biotop_dir_new'
biotop_file = '/Users/flo/Proj/sat_hedge_detection/out_10_01_21_size12.csv'

with open(biotop_file, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    notzero_count = 0
    for row in csv_reader:
        elements = row['Biotop;percentage;nPatches;']
        elements = elements.split(";")
        if elements[2] != 0:
            notzero_count = notzero_count+1

percentage = np.zeros(notzero_count)
nPatches = np.zeros(notzero_count)
bio = {}
with open(biotop_file, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        elements = row['Biotop;percentage;nPatches;']
        elements = elements.split(";")
        bio[line_count] = elements[0]
        percentage[line_count] = elements[1]
        nPatches[line_count] = elements[2]
        line_count = line_count +1


for filename in os.listdir(source_directory):
    if filename != '.DS_Store':
        bio_path = os.path.join(source_directory, filename)
    else:
        continue

    for dict_i in bio:
        if bio[dict_i] == filename:
            temp_percent = percentage[dict_i]

    #create new bio folder
    target_bio = target_directory + '/' + filename
    if not os.path.exists(target_bio):
        os.makedirs(target_bio)

    for bio_image in os.listdir(bio_path):
        bio_image_path = os.path.join(bio_path, bio_image)
        #print(bio_image_path)

        if bio_image.startswith("1_") or bio_image.startswith("2_"):
            print("Copy (1/2): ",bio_image)
            copyfile(bio_image_path, target_bio+'/'+bio_image)

            old_file = target_bio+'/'+bio_image
            new_file = os.path.join(target_bio+'/'+ str(int(temp_percent)) + '_' + bio_image)
            os.rename(old_file, new_file)

        if bio_image.startswith("B_"):
            print("Copy (B): ",bio_image)
            copyfile(bio_image_path, target_bio+'/'+bio_image)

            old_file = target_bio+'/'+bio_image
            new_file = os.path.join(target_bio+'/'+ str(int(temp_percent)) + '_' + bio_image)
            os.rename(old_file, new_file)

        if bio_image.endswith(".txt"):
            print("Copy (text): ",bio_image)
            copyfile(bio_image_path, target_bio+'/'+bio_image)

            #add sagis link
            txt_file = open(target_bio+'/'+bio_image, 'r')
            lines = txt_file.readlines()
            line_count = 0
            for line in lines:
                if line_count == 10:
                    gps_line = line.strip()[17:]
                    gps = gps_line.split(',')
                    #print(gps[0])
                    y,x = transformer.transform(gps[0], gps[1])
                    #print(transformer.transform(47.96569412118293,13.221233099780582))
                line_count +=1

            sagis_link = 'https://www.salzburg.gv.at/sagisonline/init.aspx?hotspot=landsbg|default|1:1000|'+str(x)+'|'+str(y)+'|hotspot0.gif|&redliningid=3fxndyaiszcn2fxvzgjb0zyg'
            txt_file = open(target_bio+'/'+bio_image, 'a')
            txt_file.write("Sagis-link: "+ sagis_link)
            txt_file.close()

    #break
