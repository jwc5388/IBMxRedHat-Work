import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import random

SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(tf.__version__)

# x = np.array([1,2,3,4,5])
# y = np.array([1,2,3,4,5])

x = np.array([1])
y = np.array([1])

model = Sequential()
model.add(Dense(3,input_dim = 1))
model.add(Dense(2))
model.add(Dense(1))

#########################동결##########################
# model.trainable = False #동결
model.trainable = True  #동결 x

print('=============================================')
print(model.weights)
print('=============================================')



#3 compile, train

model.compile(loss = 'mse', optimizer= 'adam')
model.fit(x,y, batch_size=1, epochs=200, verbose=0)

#4 evaluate, predict
y_pred = model.predict(x)
print(y_pred)


#x=1 y=1 가중치 동결 후 훈련
############손계산 해라


#동결하지 않고 훈련한거 손계산할것

print('=============================================')
print(model.weights)
print('=============================================')
