#copied from 18-3
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import time

#1 data
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) #(442, 10) (442,)

# exit()

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.75, random_state=333)

# model = Sequential()
# model.add(Dense(64, input_dim=10, activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64))
# model.add(Dropout(0.3))
# model.add(Dense(32))
# model.add(Dense(1))  # regression output


input1 = Input(shape=(10,))
dense1 = Dense(64)(input1) #summary 의 이름을 바꿔줄 수 있다
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(64)(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(64)(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(32)(drop3)
output1 = Dense(1)(dense4)

model = Model( inputs = input1, outputs = output1)


model.compile(loss ='mse',optimizer = 'adam', metrics=['acc'])


start = time.time()
##list 형식으로 저장. 매 epoch 끝날 때마다 하나씩 들어가니, epoch 갯수만큼 
hist = model.fit(x_train,y_train, epochs = 200, batch_size =8,
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

# loss: 3004.5361328125
# accuracy: 0.0

# loss: 2950.740234375
# accuracy: 0.0


# cpu = 18.287519931793213

# gpu = 28.71335220336914