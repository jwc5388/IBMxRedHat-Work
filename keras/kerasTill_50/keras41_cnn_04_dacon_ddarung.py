
#https://dacon.io/competitions/open/235576/overview/description

import numpy as np      #1.23.0
import pandas as pd     #2.2.3

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time 
#1 data
# train_csv = pd.read_csv('.\_data\dacon\따릉이\train.csv')
## get rid of index column and just get the data

path = 'Study25/_data/dacon/ddarung/'
path_save = 'Study25/_data/dacon/ddarung/csv_files/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
# print(train_csv) #[1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
# print(test_csv) #[715 rows x 9 columns]

submission_csv = pd.read_csv(path+ 'submission.csv', index_col=0)
# print(submission_csv) # [715 rows x 1 columns]

train_csv = train_csv.dropna()


# test_csv = test_csv.dropna()
test_csv = test_csv.fillna(train_csv.mean())
print(test_csv.info())

x = train_csv.drop(['count'], axis=1)  #count라는 axis=1 열 삭제, 행은 axis =0
print(x) #[1459 rows x 9 columns]
# y = train_csv.

#take only the count colum to y 
y = train_csv['count'] 
print(y) #(1459,)



x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state= 3333)

# x_train = x_train.reshape(-1, 3,3,1)
# x_test = x_test.reshape(-1,3,3,1)

# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
x_train = x_train.to_numpy().reshape(-1, 3, 3, 1)
x_test = x_test.to_numpy().reshape(-1, 3, 3, 1)
y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)


# 4. Build Model
model = Sequential()

model.add(Conv2D(64, (2,2), input_shape=(3,3, 1), activation='relu', padding='same'))
model.add(BatchNormalization())
# model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
model.add(BatchNormalization())
# model.add(MaxPooling2D())
model.add(Dropout(0.3))

# model.add(Conv2D(128, (3, 3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D())
# model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))

model.summary()

# exit()

model.compile(loss = 'mse', optimizer= 'adam', metrics= ['mae'])


start = time.time()
##list 형식으로 저장. 매 epoch 끝날 때마다 하나씩 들어가니, epoch 갯수만큼 
hist = model.fit(x_train,y_train, epochs = 200, batch_size =32,
          verbose = 1,
          validation_split = 0.2,
          
          )
end = time.time()

loss, mae = model.evaluate(x_test, y_test)
print('loss:', loss)
print('mae:', mae)
print('걸린시간:', end - start)


# loss: 2625.902099609375
# mae: 36.87104797363281
# 걸린시간: 39.89507508277893