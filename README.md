# Biotop_detection

Automated Biotop processing using public availiable aerial image data (orthophotos via wmts layers) and geospatial vector data (shape-file) for image generation/download and processing/classificatin via Tensforlow Inception net

- Download Biotops based on wmts and biotio-shape file
- Re-Training of Inception v4 Net with biotop samples (Transfer-Learing)
- Classsification of biotops with trained net using a sliding window

# Modules:
## Image generation:

- Dependencies:
```
import geopandas as gpd
import folium
from folium.folium import Map
from pyproj import Transformer
from selenium import webdriver
import os
import io
import numpy as np
from PIL import Image, ImageFilter
import cv2
```
- Driver chrome (firefox also possible):

https://chromedriver.chromium.org/downloads
```
copy to usr/local/bin/
```


## net train
- Run training xxxnet
```
make_image_classifier   --image_dir /Users/flo/Desktop/backup_output_biotop/training_patches/   --tfhub_module https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4   --image_size 224   --saved_model_dir my_dir/new_model   --labels_output_file class_labels.txt   --tflite_output_file new_mobile_model.tflite --train_epochs 20
```
- Run traing inception_v3
```
make_image_classifier   --image_dir /Users/flo/Desktop/backup_output_biotop/training_patches/   --tfhub_module https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4   --image_size 80   --saved_model_dir my_dir/new_model   --labels_output_file class_labels.txt   --tflite_output_file new_mobile_model.tflite --train_epochs 20
```
- Run classifiction and sliding window
```
python iterate_image_patches.py   --input_mean 0 --input_std 255   --model_file new_mobile_model.tflite --label_file class_labels.txt --image /Users/flo/Proj/sat_hedge_detection/data/_temp_sliding_window/im/32.png
```
