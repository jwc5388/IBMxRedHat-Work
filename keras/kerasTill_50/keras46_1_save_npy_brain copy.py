import numpy as np
import pandas as pd
import time
import tensorflow as tf
from datetime import datetime
from keras.models import Sequential, Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
    rescale=1/255., #scaling 0~255.
    horizontal_flip= True,
    vertical_flip= True,
    width_shift_range= 0.1,
    height_shift_range=0.1,
    rotation_range= 5,
    zoom_range= 1.2,
    shear_range=0.7,
    
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(
    rescale=1/255.,
    horizontal_flip=True,
    vertical_flip= True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
)


path_train = '/workspace/TensorJae/Study25/_data/brain/train/'
path_test = '/workspace/TensorJae/Study25/_data/brain/test'
np_path = '/workspace/TensorJae/Study25/_save/save_brain_npy/'


start = time.time()

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size= (150,150),
    batch_size= 100,
    class_mode = 'binary',
    color_mode= 'grayscale',
    shuffle = True,
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size= (150,150),
    batch_size=100,
    class_mode= 'binary',
    color_mode= 'grayscale',
    shuffle=True,
)

# ✅ 배치 데이터 합치기 (Combine batch data into one full dataset)
all_x = []  # List to store all batches of image data
all_y = []  # List to store all batches of label data

# Loop through all batches in the training data
for i in range(len(xy_train)):
    x_batch, y_batch = xy_train[i]  # Get one batch of images and labels
    all_x.append(x_batch)           # Add image batch to the list
    all_y.append(y_batch)           # Add label batch to the list

# Concatenate all batches into a single NumPy array
# Shape becomes (total_images, 150, 150, 1)
x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)

# Print the final shapes to verify the dataset size
print('x.shape:', x.shape, 'y.shape:', y.shape)

# ✅ 저장 (Save the full dataset as .npy files for faster future loading)
np.save(np_path + 'keras46_x_train.npy', x)  # Save training images
np.save(np_path + 'keras46_y_train.npy', y)  # Save training labels


all_z = []
for i in range(len(xy_test)):
    x_batch, y_batch = xy_test[i]
    all_z.append(x_batch)
end2 = time.time()

z = np.concatenate(all_z, axis=0)

start2= time.time()
np.save(np_path + 'keras_x_test.npy', arr=z)

end3 = time.time()

