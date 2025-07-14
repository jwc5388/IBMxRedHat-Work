from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#70% train, %30 evaluation
#1 data
# x = np.array([1,2,3,4,5,6,7,8,9,10]).T
# y = np.array([1,2,3,4,5,6,7,8,9,10]).T

# print(x.shape)
# print(y.shape)
# exit()

#training data
x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

#evaluation/testing data
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2 model
model = Sequential()
model.add(Dense(1, input_dim= 1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

# compile, train

model.compile(loss='mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=500, batch_size = 1)


loss = model.evaluate(x_test,y_test)
result = model.predict([11])

print("loss:", loss)
print("prediction of [11]:", result)