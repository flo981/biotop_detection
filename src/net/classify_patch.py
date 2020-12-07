import itertools
import os
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2


def get_class_string_from_index(index):
   for class_string, class_index in valid_generator.class_indices.items():
      if class_index == index:
         return class_string


module_selection = ("inception_v3", 299) #@param ["(\"mobilenet_v2_100_224\", 224)", "(\"inception_v3\", 299)"] {type:"raw", allow-input: true}
handle_base, pixels = module_selection
#MODULE_HANDLE ="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
#print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

BATCH_SIZE = 16 #@param {type:"integer"}

data_dir = '../data/training_patches/'

datagen_kwargs = dict(rescale=1./255, validation_split=.20)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                   interpolation="bilinear")


#test_image = Image.open('../data/training_patches/bio/1_bio.png')
#image3 = Image.open('../data/training_patches/bio/1_bio.png').convert('RGB')
#image3 = image3.resize((299,299), resample= 2) #2:Image.BILINEAR
#image3 = np.asarray(image3)

#################################################################### again with test_datagen
print('### TESTING ###')
#test_data_dir = '../data/tf_input_patches/'
test_data_dir = '../data/_temp_sliding_window/'

generated_test_data_dir = '../data/tf_output_patches/'
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
test_generator = test_datagen.flow_from_directory(
    test_data_dir ,shuffle=False, **dataflow_kwargs) #, save_to_dir=generated_test_data_dir, save_format='png'

# count=0
# for item in next(test_generator):
#     count = count+1
#     print(count)
#     #print(len(item),type(item)) = 2 <class 'tuple'>
#     test = item[0]
#
#     imgplot = plt.imshow(test)
#     plt.show()
model = tf.keras.models.load_model('/Users/flo/Desktop/saved_biotop_model_v1')

i = 0
for i in range(0,100):
    i=i+1
    print(i)
    x, y = next(test_generator,-1)
    #test_image = x[0, :, :, :]
    image3 = Image.open(test_data_dir+'im/9.png').convert('RGB')
    image3 = image3.resize((299,299), resample= 2) #2:Image.BILINEAR
    image3 = np.asarray(image3)
    test_image = image3
    #true_index = np.argmax(y[0])
    cv2.imshow("window", test_image)
    key = cv2.waitKey(0)



    #image3 = Image.open('../data/tf_output_patches/_7_9406846.png')
    #test_image = np.asarray(image3)
    #print(test_image.shape)



    # # Expand the validation image to (1, 224, 224, 3) before predicting the label
    prediction_scores = model.predict(np.expand_dims(test_image, axis=0))
    #prediction_scores = model.predict(image3)
    print("Predicted scores: ", prediction_scores)
    predicted_index = np.argmax(prediction_scores)
    if predicted_index==0:
        print("bio")
    else:
        print("no bio")
    #print("True label: " + get_class_string_from_index(true_index))
    #print("Predicted label: " + get_class_string_from_index(predicted_index))
    print("-----------------------")
