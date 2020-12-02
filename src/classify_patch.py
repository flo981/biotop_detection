import itertools
import os
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image



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

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

do_data_augmentation = True #@param {type:"boolean"}
if do_data_augmentation:
  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rotation_range=40,
      horizontal_flip=True,
      width_shift_range=0.2, height_shift_range=0.2,
      shear_range=0.2, zoom_range=0.2,
      **datagen_kwargs)
else:
  train_datagen = valid_datagen
train_generator = train_datagen.flow_from_directory(
    data_dir, subset="training", shuffle=True, **dataflow_kwargs)

do_fine_tuning = True #@param {type:"boolean"}

#image2 = Image.open('../data/training_patches/bio/1_bio.png')
image3 = Image.open('../data/training_patches/bio/1_bio.png').convert('RGB')
#image3 = image3.resize((299,299), resample= 2) #2:Image.BILINEAR
image3 = np.asarray(image3)

#
# data_dir2 = '../data/input_patches/'
# valid_datagen2 = tf.keras.preprocessing.image.ImageDataGenerator(
#     **datagen_kwargs)
# valid_generator2 = valid_datagen2.flow_from_directory(
#     data_dir2, subset="validation", shuffle=False, **dataflow_kwargs)
#
model = tf.keras.models.load_model('/Users/flo/Desktop/saved_biotop_model_v2')

#
# #image2 = image2[0, :, :, :]
# x, y = next(valid_generator2)
# image2 = x[0, :, :, :]


x, y = next(valid_generator)
image = x[0, :, :, :]
print(image.shape)
#print(image2.shape)
print(image3.shape)
#true_index = np.argmax(y[0])

plt.imshow(image3)
plt.axis('off')
plt.show()

plt.imshow(image)
plt.axis('off')
plt.show()


# Expand the validation image to (1, 224, 224, 3) before predicting the label
prediction_scores = model.predict(np.expand_dims(image3, axis=0))
#prediction_scores = model.predict(image3)
print(prediction_scores)
# predicted_index = np.argmax(prediction_scores)
# print("True label: " + get_class_string_from_index(true_index))
# print("Predicted label: " + get_class_string_from_index(predicted_index))
