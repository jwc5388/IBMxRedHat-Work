import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. data
x = np.array(range(1,17))
y = np.array(range(1,17))

#train_test_split으로 나눠라 10:3:3

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.85,
                                                    shuffle=True, random_state=33)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, train_size=0.8,
                                                    shuffle=True, random_state=33)

# x_train, x_temp, y_train, y_temp = train_test_split(x,y, train_size=10/17, random_state=33)
# x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, train_size=4/7, random_state=33)
# print(x_train)
# print(x_val)
# print(x_test)
# exit()

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