_____________________________________________________________________________
### Todos:
folium2png

- ~~biotop-type selection (current only 8.1.1.1+8.1.1.2)~~
- text file generation
- folium2png -> input which png files to generate
- resize output

training
- new training dataset
- patch size training <> patch size training data ?! 
- more classes?!
 
net
- sliding window to squre meter calcuation
- better outfile (excel after list is filled)
- total % 

  
_____________________________________________________________________________

# Biotope detection

Automated biotope processing using public availiable aerial image data (orthophotos via wmts layers) and geospatial vector data (shape-file) for image generation/download and processing/classificatin via Tensforlow Inception net

- Download biotopes based on wmts and biotope-shape file
- Re-Training of Inception v4 Net with biotope samples (Transfer-Learing)
- Classsification of biotops with trained net using a sliding window

![alt text](https://github.com/flo981/biotop_detection/blob/net_v2/Examples/images/big.png?raw=true)
![alt text](https://github.com/flo981/biotop_detection/blob/net_v2/Examples/images/small.png?raw=true)



# Modules:
## Image generation:

- Dependencies:
```
import os
import io
from PIL import Image
import geopandas as gpd
import folium
from folium.folium import Map
from folium.features import DivIcon
from pyproj import Transformer
```
- chrome driver (safari also possible):
https://chromedriver.chromium.org/downloads
- firefox driver:
https://github.com/mozilla/geckodriver/releases
```
copy to usr/local/bin/
```

Jupyter Notebook Example in ```src/bio_generator/```

## net train

- Dependencies:
```
import cv2
import os
from shutil import copyfile
import numpy as np
import imutils
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf  # TF2
```

- Run training mobilenet_v2
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
