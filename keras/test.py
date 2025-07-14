# # from re import X
# # from tkinter import Y
from os import times
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))


# # from random import shuffle
# # from keras.datasets import fashion_mnist
# # from keras.preprocessing.image import ImageDataGenerator
# # from keras.models import Sequential
# # from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation
# # from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# # import numpy as np
# # from sklearn.metrics import accuracy_score
# # import time
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# # x_train = x_train/255.
# # x_test = x_test/255.

# # datagen = ImageDataGenerator(
# #     rescale=1/255.,
# #     horizontal_flip=True,
# #     vertical_flip= True,
# #     width_flip =True,
# #     rotation_range= 15,
    
# # )


# # augment_size = 40000
# # randidx = np.random.randint(x_train.shape[0], size = augment_size)
# # randidx = np.random.randint(60000, 40000)

# # #random index 40000 out of 60000
# # # print(randidx) -> 4만개의 데이터를 줌

# # x_augmented = x_train[randidx].copy()
# # y_augmented = y_train[randidx].copy()

# # x_augmented = x_augmented.reshape(-1,28,28,1)

# # x_augmented = datagen.flow(
# #     x_augmented,
# #     y_augmented,
# #     batch_size= augment_size,
# #     shuffle= False,
# #     save_to_dir=''
# # ).next()[0]


# # x_train = x_train.reshape(60000, 28, 28, 1)
# # x_test = x_test.reshape(-1,28,28,1)
# # print(x_train.shape)


# # x_train = np.concatenate((x_train, x_augmented))
# # y_train = np.concatenate((y_train, y_augmented))


# # y_train = pd.get_dummies(y_train)
# # y_test = pd.get_dummies(y_test)


# # model = Sequential()

# # model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(28,28,1)))

# import numpy as np
# import time

# from keras.models import Sequential
# from keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout, BatchNormalization

# datasets = np.array([1,2,3,4,5,6,7,8,9,10]) #can be sequential or not. 종속되지 마라

# x = np.array([[1,2,3],
#               [2,3,4],
#               [3,4,5],
#               [4,5,6],
#               [5,6,7],
#               [6,7,8],
#               [7,8,9],
#               ])

# y = np.array([4,5,6,7,8,9,10])

# #timestep 클수록 잘 맞을 수도 있다/.
# #time_step은 중복될 수 있다.

# print(x.shape, y.shape) #(7, 3) (7,)

# x = x.reshape(x.shape[0], x.shape[1], 1)

# #(7,3,1) -> (batch_size, time_steps, feature)


# model = Sequential()
# model.add(SimpleRNN(10, input_shape = (3,1), activation= 'relu'))
# model.add(Dense(5,activation='relu'))
# model.add(Dense(1))
# model.summary()




# model.compile(loss = 'mse', optimizer = 'adam')

# start = time.time()

# model.fit(x,y, epochs= 3000)

# end = time.time()

# loss = model.evaluate(x,y)
# print('loss:', loss)
# x_pred = np.array([8,9,10]).reshape(-1,3,1)
# result = model.predict(x_pred)
# print('[8,9,10]의 결과:', result)

# print('time:', end-start)




import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout, BatchNormalization


#1 data
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

# 시계열 데이터라면 time_step에 따라 잘라내야함
x = np.array([[1,2,3],  # 다음은 4야~
              [2,3,4],  # 다음은 5야~
              [3,4,5],  # 다음은 6이야~
              [4,5,6],  # 다음은 7이야~
              [5,6,7],  # 다음은 8이야~
              [6,7,8],  # 다음은 9야~
              [7,8,9],  # 다음은 10이야~ -> 8,9,10 다음은 뭐게~? -> 이런 게 RNN의 작동 목적
             ])

y = np.array([4,5,6,7,8,9,10])
x = x.reshape(x.shape[0], x.shape[1], 1)

model = Sequential()

model.add(SimpleRNN(10, input))


import numpy as np
a = np.array(range(1,101))
x_pred = np.array(range(96,106))

timesteps = 6

def split_1d(dataset, timesteps):
    all = []
    for i in range(len(dataset- timesteps+1)):
        subset = dataset[i: (i+timesteps)]
        all.append(subset)
        
    all = np.array(all)
    x = all[:,:-1]
    y = all[:,-1]
    return x,y


x,y = split_1d(a,6)


def split_all(dataset, timesteps):
    all = []
    for i in range(len(dataset) - timesteps +1):
        subset = dataset[i : (i+timesteps)]
        all.append(subset)
    all = np.array(all[:])
    return all
x_pred = split_all(x_pred, 5)

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout, BatchNormalization

datasets = np.array([1,2,3,4,5,6,7,8,9,10,11])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              [8,9,10]])

y = np.array([[4,5,6,7,8,9,10,11]])

x = x.reshape(x.shape[0], x.shape[1], 1)


model = Sequential()

model.add(SimpleRNN(10, input_shape=(3,1)))

model.add(Dense(128))
model.add(Dense(1))


