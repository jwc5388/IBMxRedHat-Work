from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1 data
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])


#2 model
model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(6,input_dim=5))
model.add(Dense(6,input_dim=6))
model.add(Dense(2,input_dim=6))
model.add(Dense(1,input_dim=2))


#3 compile, training

epochs=300

model.compile(loss='mse', optimizer = 'adam')
model.fit(x,y,epochs=epochs)

#evaluate
loss = model.evaluate(x,y)
print("#######################################")
print('epochs: ', epochs)
print('loss: ', loss)

results = model.predict([6])
print('6 prediction: ', results)