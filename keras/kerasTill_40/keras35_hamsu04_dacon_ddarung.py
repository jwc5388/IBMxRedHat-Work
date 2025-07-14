
#https://dacon.io/competitions/open/235576/overview/description

import numpy as np      #1.23.0
import pandas as pd     #2.2.3

from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Dropout, Input
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

model = Sequential()
model.add(Dense(77, input_dim =9, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(77, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(77))
model.add(Dropout(0.3))
model.add(Dense(77))
model.add(Dense(77))
model.add(Dense(1))

input1 = Input(shape=(9,))
dense1 = Dense(77)(input1) #summary 의 이름을 바꿔줄 수 있다
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(77)(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(77)(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(77)(drop3)
dense5 = Dense(77)(dense4)
output1 = Dense(1)(dense5)

model1 = Model( inputs = input1, outputs = output1)


#compile and train
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])


start = time.time()
##list 형식으로 저장. 매 epoch 끝날 때마다 하나씩 들어가니, epoch 갯수만큼 
hist = model.fit(x_train,y_train, epochs = 100, batch_size =4,
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

#Droupout 
# loss: 3063.2607421875
# accuracy: 0.007518797181546688

# cpu = 66.3422782421112

# gpu = 