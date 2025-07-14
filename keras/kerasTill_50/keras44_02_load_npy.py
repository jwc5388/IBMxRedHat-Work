import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Input, Dropout, BatchNormalization, Activation, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import tensorflow as tf

import time

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Input, Dropout, BatchNormalization, Activation, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator

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

path_train = '/home/jupyter/TensorJae/Study25/_data/cat_dog/train2/'
path_test = '/home/jupyter/TensorJae/Study25/_data/cat_dog/test2/'
# path_train = 'Study25/_data/kaggle/cat_dog/train2/'
# path_test = 'Study25/_data/kaggle/cat_dog/test2/'
# from pathlib import Path

# base = Path.home() / "Desktop" / "IBM x RedHat" / "Study25" / "_data" / "cat_dog"
# path_train = str(base / "train2")
# path_test = str(base / "test2")
start = time.time()
xy_train = train_datagen.flow_from_directory(
    path_train,                         #경로
    target_size = (200,200),            #리사이즈, 사이즈 규격 일치, 큰거는 축소, 작은거는 확대
    batch_size = 100,                    #
    class_mode = 'binary',
    color_mode = 'rgb',
    shuffle = True,    
    classes=['cat', 'dog'],#default = False
    seed = 333,
)


xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(200, 200),
    batch_size=160,
    class_mode='binary',
    color_mode='rgb',
    classes=['cat', 'dog'],

)
print(xy_train[0][0].shape)
print(xy_train[0][1].shape)
print(len(xy_train))
end = time.time()
# #######모든 수치화된 batch 데이터를 하나로 합치기#########
all_x = []
all_y = []

for i in range(len(xy_train)):
    x_batch, y_batch = xy_train[i]
    all_x.append(x_batch)
    all_y.append(y_batch)

    
end2 = time.time()
# print(all_x)

x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)

print('x.shape:', x.shape)
print('y.shape:', y.shape)


start2 = time.time()
# path_train = 'Study25/_data/kaggle/cat_dog/train2/'
# path_test = 'Study25/_data/kaggle/cat_dog/test2/'
np_path = 'Study25/_save/save_npy/'
# np_path = '/home/jupyter/TensorJae/Study25/_save/save_npy/'
np.save(np_path + 'keras44_x_train.npy', arr =x)
np.save(np_path + 'keras44_y_train.npy', arr =y)

end3 = time.time()


print('time1:', round(end-start,2), 'seconds')
print('time2:', round(end2-end,2), 'seconds')

print("npy 저장시간:", round(end3- start2, 2), 'seconds')
# npy 저장시간: 4.36 seconds
exit()
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]

# x_train = xy_train[0][0]
# y_train = xy_train[0][1]

# x_test = xy_test[0][0]
# y_test = xy_test[0][1]

# print(x_train.shape, y_train.shape) #(160, 200, 200, 1) (160,)
# print(x_test.shape, y_test.shape)   #(120, 200, 200, 1) (120,)

# Found 25000 images belonging to 2 classes.
# Found 12500 images belonging to 2 classes.
# (160, 200, 200, 1) (160,)
# (160, 200, 200, 1) (160,)
# exit()
# aaa = 7
# print(y_train[aaa])

# import matplotlib.pyplot as plt

# plt.imshow(x_train[aaa], 'twilight_shifted')
# plt.show()

# x_train = x_train/255.
# x_test = x_test/255.


# print(xy_train.class_indices)  # Should show 2 classes like {'normal': 0, 'abnormal': 1}
# print(xy_train.samples)        # Should be > 160 ideally
# print(xy_test.samples)         # Should match expected test images

# exit()

model = Sequential()
model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 input_shape = (200,200,3),
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
    mode='min',
    patience=20,
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