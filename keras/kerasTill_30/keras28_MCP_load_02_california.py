import sklearn as sk # 1.1.3
import tensorflow as tf # 2.9.3
import ssl
import urllib.request
import numpy as np
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model



# ssl._create_default_https_context = ssl._create_unverified_context


#1 data

dataset= fetch_california_housing()
x = dataset.data
y = dataset.target

print(x.shape) #(20640,8)
print(y.shape) # (20640,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=333)

path = 'Study25/_save/keras28_mcp/01_boston'

model = load_model(path + 'keras28_boston_save.h5')



#compile and train
model.compile(loss = 'mse', optimizer = 'adam')


#evaluate and predict
loss = model.evaluate(x_test, y_test)
print("loss:", loss)
result = model.predict(x_test)
# print("prediction:", result)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, result)
print("r2 score:",  r2)


