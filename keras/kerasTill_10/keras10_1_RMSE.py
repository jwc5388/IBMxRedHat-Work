import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1 data

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13, 8,14,15, 9, 6,17,23,21,20])


#2 model
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state=3)

model = Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(20, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(80))
model.add(Dense(20))
model.add(Dense(1))

#3 compile and train
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1000, batch_size = 1)

#4 evaluate and predict
#evaluated by 'mse'
loss = model.evaluate(x_test, y_test)
print("loss:", loss)
result = model.predict([x_test])
print("prediction of [x]:", result)

from sklearn.metrics import r2_score, mean_squared_error

def RMSE(y_test , y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
rmse = RMSE(y_test, result)
print('RMSE:' ,rmse)

# prediction of [x]: [[14.525895 ]
#  [ 2.944673 ]
#  [ 1.9798601]
#  [17.421425 ]]
# RMSE: 3.9613803713751485
