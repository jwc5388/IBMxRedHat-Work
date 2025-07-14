import sklearn as sk # 1.1.3
import tensorflow as tf # 2.9.3
import ssl
import urllib.request
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# ssl._create_default_https_context = ssl._create_unverified_context


#1 data

dataset= fetch_california_housing()
x = dataset.data
y = dataset.target

print(x.shape) #(20640,8)
print(y.shape) # (20640,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=3)

#2 Model
model = Sequential()
model.add(Dense(128,input_dim = 8))
model.add(Dense(128))
model.add(Dense(128))
# model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))


#compile and train
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 500, batch_size = 32)

#evaluate and predict
loss = model.evaluate(x_test, y_test)
print("loss:", loss)
result = model.predict(x_test)
# print("prediction:", result)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, result)
print("r2 score:",  r2)



# loss: 0.6029499173164368
# r2 score: 0.5431547428708206

# loss: 0.6033823490142822
# r2 score: 0.5437945162331175

# loss: 0.6028112173080444
# r2 score: 0.5442262003693807