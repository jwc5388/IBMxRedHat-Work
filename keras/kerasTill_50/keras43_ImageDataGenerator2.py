import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Input, Dropout, BatchNormalization, Activation, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import tensorflow as tf

import time


train_datagen = ImageDataGenerator(
    rescale  = 1/255.  ,     #0~255 스케일링,
    # horizontal_flip = True,  #수평 뒤집기 <- 데이터 증폭 또는 변환
    # vertical_flip = True, #수직 뒤집기 <- 데이터 증폭 또는 변환
    # width_shift_range = 0.1,#평행이동 10%
    # height_shift_range = 0.1,
    # rotation_range = 5,
    # zoom_range = 1.2,
    # shear_range = 0.7,  #좌표 하나 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(찌부 만들기)
    # fill_mode = 'nearest'
    
    
)


test_datagen = ImageDataGenerator(
    rescale = 1/255. ,
    
)

# path_train = '/home/jupyter/TensorJae/Study25/_data/brain_clean/train/'
# path_test = '/home/jupyter/TensorJae/Study25/_data/brain/test/'
path_train = 'Study25/_data/brain/train/'
path_test = 'Study25/_data/brain/test/'


xy_train = train_datagen.flow_from_directory(
    path_train,                         #경로
    target_size = (200,200),            #리사이즈, 사이즈 규격 일치, 큰거는 축소, 작은거는 확대
    batch_size = 160,                    #
    class_mode = 'binary',
    color_mode = 'grayscale',
    # shuffle = True,                   #default = False
    
)


xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(200, 200),
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale',
    classes=['ad', 'normal'],  # ✅ THIS IGNORES .ipynb_checkpoints
)

x_train = xy_train[0][0]
y_train = xy_train[0][1]

x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape) #(160, 200, 200, 1) (160,)
print(x_test.shape, y_test.shape)   #(120, 200, 200, 1) (120,)

aaa = 7
print(y_train[aaa])

import matplotlib.pyplot as plt

plt.imshow(x_train[aaa], 'twilight_shifted')
plt.show()

# x_train = x_train/255.
# x_test = x_test/255.


# print(xy_train.class_indices)  # Should show 2 classes like {'normal': 0, 'abnormal': 1}
# print(xy_train.samples)        # Should be > 160 ideally
# print(xy_test.samples)         # Should match expected test images

# exit()

model = Sequential()
model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 input_shape = (200,200,1),
                 activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(filters = 256,
                 kernel_size=(3,3),
                 activation='relu',
                 ))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))


model.add(Conv2D(filters = 256,
                 kernel_size=(3,3),
                 activation='relu',
                 ))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(filters = 256,
                 kernel_size=(3,3),
                 activation='relu',
                 ))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=1000,
    restore_best_weights=True
)

start = time.time()

hist = model.fit(x_train, y_train, batch_size=64,
          epochs=300,
          callbacks=[es],
          validation_split=0.2,
          verbose=1)


end = time.time()

loss, acc = model.evaluate(x_test, y_test)
result = model.predict(x_test)

print('loss:', loss)
print('acc:', acc)


print('time:', end- start)