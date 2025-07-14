#copy from 18-2

import sklearn as sk # 1.1.3
import tensorflow as tf # 2.9.3
import ssl
import urllib.request
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# ssl._create_default_https_context = ssl._create_unverified_context


#1 data

dataset= fetch_california_housing()
x = dataset.data
y = dataset.target

print(x.shape) #(20640,8)
print(y.shape) # (20640,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=333)

#2 Model
model = Sequential()
model.add(Dense(128,input_dim = 8, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128))
# model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))


#compile and train
model.compile(loss = 'mse', optimizer = 'adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 15,
    restore_best_weights = True
)
hist = model.fit(x_train, y_train, epochs = 200, batch_size = 32,
          validation_split = 0.2,
          callbacks = [es],)



import matplotlib.pyplot as plt
plt.figure(figsize=(9,6)) # 9x6 size 
plt.plot(hist.history['loss'], c = 'red', label = 'loss') # y값만 넣으면 시간순으로 그림 그림
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss')
plt.title("California loss")
plt.xlabel('epoch')
plt.ylabel('loss') 
plt.legend(loc='upper right') #우측 상단에 label 표시
plt.grid() #격자 표시


plt.show()




# #evaluate and predict
# loss = model.evaluate(x_test, y_test)
# print("loss:", loss)
# result = model.predict(x_test)
# # print("prediction:", result)

# from sklearn.metrics import r2_score

# r2 = r2_score(y_test, result)
# print("r2 score:",  r2)
