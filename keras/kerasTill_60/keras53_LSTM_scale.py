import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU
from keras.callbacks import ModelCheckpoint

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],[20,30,40],
             [30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

print(x.shape, y.shape)     #(13, 3) (13,)

x = x.reshape(x.shape[0],x.shape[1],1)
x_pred = x_pred.reshape(1,3,1)

model = Sequential()
model.add(SimpleRNN(128, input_shape=(3,1)))
model.add(Dense(64))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics='acc')

path = './_save/keras53/mcp'
model.fit(x, y, epochs=2000,)

loss = model.evaluate(x, y)
results = model.predict(x_pred)

print('results :', results)
print('Loss:', loss[0])
print('Acc :', loss[1])






