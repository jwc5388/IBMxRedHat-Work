#실습
#100,100,3 이미지를 
#10,10,11 으로 줄여봐!!

from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Input, Dropout, MaxPooling2D

model = Sequential()

model.add(Conv2D(filters = 11, kernel_size= (3,3), input_shape = (100,100,3),
                 strides=2,
                 padding='valid'))
model.add(Conv2D(filters= 11, kernel_size=(3,3),
                 strides=2,
                 padding='valid'))
model.add(Conv2D(filters= 11, kernel_size=(3,3),
                 strides=2,
                 padding='valid'))
model.add(Conv2D(filters= 11, kernel_size=(2,2),
                 strides=1,
                 padding='valid'))
model.summary()