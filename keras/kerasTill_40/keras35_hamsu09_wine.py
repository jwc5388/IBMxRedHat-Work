from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import time

dataset = load_wine()
x = dataset.data
y = dataset.target

# print(x.shape, y.shape)
#(178,13), (178,)
print(np.unique(y, return_counts=True))
print(x)
print(y)

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

y = pd.get_dummies(y).values


x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, train_size=0.8, random_state=42, stratify=y)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))


model = Sequential()
model.add(Dense(32, input_dim = 13, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation = 'softmax'))

input1 = Input(shape=(13,))

dense1 = Dense(32)(input1) #summary 의 이름을 바꿔줄 수 있다

dense2 = Dense(16)(dense1)

dense3 = Dense(16)(dense2)

dense4 = Dense(16)(dense3)

output1 = Dense(3)(dense4)

model1 = Model( inputs = input1, outputs = output1)

model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics=['acc'])

start = time.time()
##list 형식으로 저장. 매 epoch 끝날 때마다 하나씩 들어가니, epoch 갯수만큼 
hist = model.fit(x_train,y_train, epochs = 100, batch_size =32,
          verbose = 1,
          validation_split = 0.2,
          
          )
end = time.time()

import tensorflow as tf
print(tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
print(gpus)


if gpus:
    print('GPU exists!')
else:
    print('GPU doesnt exist')


print("걸린시간: ", end - start )

loss, acc = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accuracy:', acc)
# cpu = 9.079880237579346

# gpu = 15.106232166290283