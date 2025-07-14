import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Input, Dropout, BatchNormalization, Activation, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import tensorflow as tf

import time


path_train = '/workspace/TensorJae/Study25/_data/brain/train/'
path_test = '/workspace/TensorJae/Study25/_data/brain/test'
np_path = '/workspace/TensorJae/Study25/_save/save_brain_npy/'

start = time.time()
# end3 = time.time()
x_train = np.load(np_path + 'keras46_x_train.npy')
y_train = np.load(np_path + 'keras46_y_train.npy')
test = np.load(np_path + 'keras_x_test.npy')
end = time.time()

print(x_train)
print(y_train[:20])
print(x_train.shape, y_train.shape)
print('test shape:', test.shape)
# [1. 1. 0. 0. 1. 0. 1. 1. 0. 0. 0. 1. 0. 1. 1. 0. 0. 0. 1. 0.]
# (160, 150, 150, 1) (160,)
# shape test: (120, 150, 150, 1)
# time: 0.01 seconds
print('time:', round(end-start, 2), 'seconds')