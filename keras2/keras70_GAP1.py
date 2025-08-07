import numpy as np
import pandas as pd

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, Dropout, Flatten
from keras.layers import GlobalAveragePooling2D

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import time

from sklearn.metrics import accuracy_score


#CNN
#1 data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

# x 를 reshape -> (60000,28,28,1)
x_train = x_train.reshape(60000,28,28,1)
# x_test = x_test.reshape(10000,28,28,1)
#위아래 똑같음 잘 봐
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)
print(x_train.shape, x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)  #(60000, 10) (10000, 10)


#2 model 
model = Sequential([
    Conv2D(filters=100,kernel_size= (2,2), strides = 1, input_shape = (28,28,1)),
    Conv2D(filters = 50, kernel_size=(2,2)),
    Conv2D(filters=30, kernel_size=(2,2), padding='same'),
    # Flatten(),
    GlobalAveragePooling2D(),
    # model.add(GlobalAveragePooling2D),
    Dense(units=16),
    Dense(units=16),
    Dense(units=10, activation='softmax'),
    
])

model.summary()
