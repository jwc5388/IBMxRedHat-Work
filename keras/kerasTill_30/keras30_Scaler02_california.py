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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler


# ssl._create_default_https_context = ssl._create_unverified_context


#1 data

dataset= fetch_california_housing()
x = dataset.data
y = dataset.target

print(x.shape) #(20640,8)
print(y.shape) # (20640,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=333)

scaler = StandardScaler()
# loss: 0.26046690344810486
# r2 score: 0.795564393341196

# scaler = MinMaxScaler()
# loss: 0.3181079924106598
# r2 score: 0.7503230529103352

# scaler = MaxAbsScaler()
# loss: 0.3814636468887329
# r2 score: 0.7005964365095416

# scaler = RobustScaler()
# loss: 0.4343244433403015
# r2 score: 0.659106828005177

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2 Model
model = Sequential()
model.add(Dense(128,input_dim = 8, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
# model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))




#compile and train
model.compile(loss = 'mse', optimizer = 'adam')

from keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights = True,
)


##list 형식으로 저장. 매 epoch 끝날 때마다 하나씩 들어가니, epoch 갯수만큼 
hist = model.fit(x_train,y_train, epochs = 100000, batch_size =32,
          verbose = 1,
          validation_split = 0.2,
          callbacks = [es],
          
          )

#evaluate and predict
loss = model.evaluate(x_test, y_test)
print("loss:", loss)
result = model.predict(x_test)
# print("prediction:", result)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, result)
print("r2 score:",  r2)


