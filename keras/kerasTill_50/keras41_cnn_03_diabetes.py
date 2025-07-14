#copied from 18-3
from keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, BatchNormalization
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

x_train = x_train.reshape(-1, 10,1,1)
x_test = x_test.reshape(-1, 10,1,1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1,1)

# 4. Build Model
model = Sequential()

model.add(Conv2D(64, (3, 1), input_shape=(10,1, 1), activation='relu', padding='same'))
model.add(BatchNormalization())
# model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 1), activation='relu', padding='same'))
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

model.compile(loss ='mse',optimizer = 'adam', metrics=['mae'])

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

# loss: 4220.03564453125
# mae: 52.13378143310547
# 걸린시간: 25.126033782958984