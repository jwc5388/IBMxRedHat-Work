from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1 data

x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#2
model = Sequential()
model.add(Dense(1,input_dim=1))

model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs = 10000) #최소의 loss와 최적의 weight


print('=========================')

#evaluate the model
loss=model.evaluate(x,y)

print('loss: ', loss)
result = model.predict([1,2,3,4,5,6,7])

print("prediction: ", result)
