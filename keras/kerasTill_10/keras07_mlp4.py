import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 data- separate train, evaluate data
x = np.array([range(10)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1],[9,8,7,6,5,4,3,2,1,0]])

print(x.shape)
print(y.shape)

#1 is 3
x = x.T     #(10,1)
y = y.T     #(10,3)

#2 model
model = Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3))


#3 compile and train
model.compile(loss='mse', optimizer = 'adam')
model.fit(x,y, epochs =300,  batch_size =1)

#4 evaluate and predict
loss = model.evaluate(x,y)
print("loss:", loss)

result = model.predict([10])

print("prediction of [10]:", result)   #prediction: [[ 1.1000002e+01 -1.2814999e-06 -9.9999976e-01]]