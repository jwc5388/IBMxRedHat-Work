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
# model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))


path_mcp = 'Study25/_save/keras28_mcp/02_california/'
model.save(path_mcp + 'keras28_california_save.h5')

#compile and train
model.compile(loss = 'mse', optimizer = 'adam')

from keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights = True,
)



# path= 'Study25/_save/keras28_mcp/01_boston'
############# mcp save file name 만들기 ##############
import datetime 
date = datetime.datetime.now()
print(date)  #2025-06-02 13:00:52.507403
print(type(date))   #<class 'datetime.datetime'>
#시간을 string으로 만들어라
date = date.strftime("%m%d_%H%M")
print(date) #0602_1305
print(type(date))   # <class 'str'>

# path= 'Study25/_save/keras27_mcp2/'

#hist 에서 제공되는 epoch, val_loss
filename = '{epoch:04d}-{val_loss:.4f}.h5'
filepath = "".join([path_mcp, 'k28_', date, '_', filename])

print(filepath)

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,
    filepath = filepath
)


##list 형식으로 저장. 매 epoch 끝날 때마다 하나씩 들어가니, epoch 갯수만큼 
hist = model.fit(x_train,y_train, epochs = 100000, batch_size =32,
          verbose = 1,
          validation_split = 0.2,
          callbacks = [es, mcp],
          
          )

#evaluate and predict
loss = model.evaluate(x_test, y_test)
print("loss:", loss)
result = model.predict(x_test)
# print("prediction:", result)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, result)
print("r2 score:",  r2)


