#CNN -> DNN

import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical

#1 data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

x_train =x_train/255.
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) #1.0 0.0
print(np.max(x_test), np.min(x_test))   #1.0 0.0

print(x_train.shape[0])#60000
print(x_train.shape[1])#28
print(x_train.shape[2])#28

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()

y_train = y_train.reshape(60000,1)
#전체 데이터 
y_test = y_test.reshape(-1,1)


print(y_train.shape, y_test.shape) #(60000, 1) (10000, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)


model = Sequential()

model.add(Dense(128, input_shape = (28*28,)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

#3 compile 

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode = 'auto',
    patience=15,
    restore_best_weights=True
)


start = time.time()

hist = model.fit(x_train, y_train, epochs= 500, batch_size=64, verbose=1,callbacks=[es])

end = time.time()



loss, acc = model.evaluate(x_test, y_test, verbose=1)

print('loss:', loss)
print('acc:', acc)

print('걸린시간:', end-start)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_test, y_pred)


# loss: 0.0805799588561058
# acc: 0.9846000075340271
# 걸린시간: 1082.6662456989288