
#!/usr/bin/env python3

import numpy as np
from PIL import Image, ImageFilter

# Open image and make into Numpy array
im = Image.open('../data/output_biotop/1_564050102.png').convert('RGB')
na = np.array(im)
orig = na.copy()    # Save original

# Median filter to remove outliers
im = im.filter(ImageFilter.MedianFilter(3))

# Find X,Y coordinates of all yellow pixels
yellowY, yellowX = np.where(np.all(na==[255,0,0],axis=2))
top, bottom = yellowY[0], yellowY[-1]
left, right = yellowX[0], yellowX[-1]
print(top,bottom,left,right)

# Extract Region of Interest from unblurred original
ROI = orig[top:bottom, left:right]

Image.fromarray(ROI).save('result.png')