import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

new_model = tf.keras.models.load_model('/Users/flo/Desktop/saved_biotop_model_v1')

# Check its architecture
test_images = './data/training_patches/no_bio/12_bio'
new_model.predict(test_images)
