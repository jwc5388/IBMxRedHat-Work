import sklearn as sk # 1.1.3
import tensorflow as tf # 2.9.3
import ssl
import urllib.request
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time

# ssl._create_default_https_context = ssl._create_unverified_context


#1 data

dataset= fetch_california_housing()
x = dataset.data
y = dataset.target

print(x.shape) #(20640,8)
print(y.shape) # (20640,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=333)

#2 Model
model = Sequential()
model.add(Dense(128,input_dim = 8, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.3))
# model.add(Dense(128))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(1))


input1 = Input(shape=(8,))
dense1 = Dense(128)(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(128)(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(64)(drop2)
drop3 = Dropout(0.2)(dense3)
output1 = Dense(1)(drop3)

model1 = Model(inputs = input1, outputs = output1)



#compile and train
model.compile(loss = 'mse', optimizer = 'adam', metrics= ['acc'])


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
# loss: 1.2742666006088257
# accuracy: 0.0026647287886589766

# cpu = 67.11176180839539

# gpu = 330.96838879585266
