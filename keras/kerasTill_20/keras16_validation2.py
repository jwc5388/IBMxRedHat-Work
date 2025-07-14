import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. data
x = np.array(range(1,17))
y = np.array(range(1,17))

#리스트의 슬라이싱으로 10:4:3으로 나눈다

x_train = x[:10]
y_train = y[:10]

x_val = x[10:14]
y_val = y[10:14]

x_test = x[14:]
y_test = x[14:]


model = Sequential()
model.add(Dense(10, input_dim = 1, activation = 'relu'))
model.add(Dense(10 , activation = 'relu'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 200, batch_size = 1,
          verbose = 1,
          validation_data = (x_val,y_val))


#evaluate and predict
loss = model.evaluate(x_test, y_test)
result = model.predict(np.array([17]))

print("loss:", loss)
print("prediction:", result)

# r2 = r2_score(y_test, result)
# print("r2 score:", r2)

# rmse = np.sqrt(mean_squared_error(y_test, result))
# print("rmse:", rmse)