from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#70% train, %30 evaluation
#1 data
x = np.array([1,2,3,4,5,6,7,8,9,10]).T
y = np.array([1,2,3,4,5,6,7,8,9,10]).T

# print(x.shape)
# print(y.shape)
# exit()

# #training data
# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])

# #evaluation/testing data
# x_test = np.array([8,9,10])
# y_test = np.array([8,9,10])

# numpy array slicing!!!!=============================
# arr = np.array([10, 20, 30, 40, 50])
# Slice	Description	Result
# arr[1:4]	Elements from index 1 to 3	[20 30 40]
# arr[:3]	First 3 elements	[10 20 30]
# arr[2:]	From index 2 to end	[30 40 50]
# arr[-2:]	Last 2 elements	[40 50]
# arr[::2]	Every 2nd element	[10 30 50]
# arr[::-1]	Reversed array	[50 40 30 20 10]

# from index 0 to 6 [:7]
x_train = x[:7]
y_train = y[:7]

#from index 7 to the end
x_test = x[7:]
y_test = y[7:]

print(x_train)
print(x_test)

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