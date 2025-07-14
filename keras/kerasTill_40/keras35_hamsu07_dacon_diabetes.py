#https://dacon.io/competitions/official/236068/overview/description


import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint

import time


path = 'Study25/_data/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col= 0)

print(train_csv.info()) # [652 rows x 9 columns]
print(test_csv.info()) #  [116 rows x 8 columns]
print(sample_submission_csv.info()) # [116 rows x 1 columns]

# print(train_csv.columns)

# exit()ㅠ ㅍ



x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())

print(x)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state= 33)


#2 model
model = Sequential()
model.add(Dense(100, input_dim = 8, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))



input1 = Input(shape=(1,))
dense1 = Dense(100)(input1) #summary 의 이름을 바꿔줄 수 있다

dense2 = Dense(100)(dense1)

dense3 = Dense(100)(dense2)

dense4 = Dense(100)(dense3)
dense5 = Dense(100)(dense4)
dense6 = Dense(100)(dense5)

output1 = Dense(1)(dense6)

model1 = Model( inputs = input1, outputs = output1)



#3 compile and train
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])



start = time.time()
##list 형식으로 저장. 매 epoch 끝날 때마다 하나씩 들어가니, epoch 갯수만큼 
hist = model.fit(x_train,y_train, epochs = 100, batch_size =32,
          verbose = 1,
          validation_split = 0.2,
          
          )
end = time.time()

import tensorflow as tf
print(tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
print(gpus)


if gpus:
    print('GPU exists!')
else:
    print('GPU doesnt exist')


print("걸린시간: ", end - start )
loss, acc = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accuracy:', acc)

# loss: 0.9071439504623413
# accuracy: 0.7099236845970154

# cpu = 10.496047973632812

# gpu = 