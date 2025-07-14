import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



#1 data
x = np.array([range(10), range(21,31), range(201,211)]) #(3,10)

y = np.array([[1,2,3,4,5,6,7,8,9,10], [10,9,8,7,6,5,4,3,2,1]]) #(2,10)

x = x.T #(10,3)
y = y.T # (10,2) ##############check notes#################!!!!!!!!!!! multiple columns possible!!!!
#2 model
model = Sequential()
model.add(Dense(10, input_dim =3))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(2))

#3 compile and train

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y, epochs = 500, batch_size =1)

#4 evaluate and predict

loss = model.evaluate(x,y)
print("Loss:", loss)

result = model.predict([[10,31,211], [11,32,212]])
print("prediction of [10,31,211, [11,32,212]:", result)

