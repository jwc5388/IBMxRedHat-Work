import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Input, Dropout, BatchNormalization, Activation
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import keras.preprocessing.image as kpi

idg = kpi.ImageDataGenerator()

train_datagen = ImageDataGenerator(
    rescale  = 1/255.  ,     #0~255 스케일링,
    horizontal_flip = True,  #수평 뒤집기 <- 데이터 증폭 또는 변환
    vertical_flip = True, #수직 뒤집기 <- 데이터 증폭 또는 변환
    width_shift_range = 0.1,#평행이동 10%
    height_shift_range = 0.1,
    rotation_range = 5,
    zoom_range = 1.2,
    shear_range = 0.7,  #좌표 하나 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(찌부 만들기)
    fill_mode = 'nearest'
    
    
)


test_datagen = ImageDataGenerator(
    rescale = 1/255. ,
    
)


path_train = 'Study25/_data/brain/train/'
path_test = 'Study25/_data/brain/test/'


xy_train = train_datagen.flow_from_directory(
    path_train,                         #경로
    target_size = (200,200),            #리사이즈, 사이즈 규격 일치, 큰거는 축소, 작은거는 확대
    batch_size = 10,                    #
    class_mode = 'binary',
    color_mode = 'grayscale',
    # shuffle = True,                   #default = False
    
)


xy_test = test_datagen.flow_from_directory(
    path_test,                         #경로
    target_size = (200,200),            #리사이즈, 사이즈 규격 일치, 큰거는 축소, 작은거는 확대
    batch_size = 10,                    #
    class_mode = 'binary',
    color_mode = 'grayscale',
    # shuffle = True,                   #default = False
    
)

# print(xy_train.shape)  ERROR tuple object has no attribute shape


print(xy_train[0])
print(len(xy_train))            #16
print(xy_train[0][0].shape)     #(10, 200, 200, 1)
print(xy_train[0][1].shape)     #(10,)
print(xy_train[0][1])           #y 열개의 label 출력 [0. 1. 0. 0. 1. 1. 1. 1. 1. 0.]
# print(xy_train[16])             # ValueError: Asked to retrieve element 16, but the Sequence has length 16
# print(xy_train[0][2])           #IndexError: tuple index out of range
print(type(xy_train))       #<class 'keras.src.legacy.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))
print(type(xy_train[0][0]))
print(type(xy_train[0][1]))



x_train = xy_train[0][0]
y_train = xy_train[0][1]
