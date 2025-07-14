
#copied from 18-5

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time

path = 'Study25/_data/kaggle/bike/'
path_save = 'Study25/_data/kaggle/bike/csv_files/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path+ 'sampleSubmission.csv', index_col=0)


# Prepare features and target
x = train_csv.drop(['casual', 'registered', 'count'], axis = 1)
y = train_csv['count']

print(f"X shape: {x.shape}")
print(f"y shape: {y.shape}")

# X shape: (10886, 8)
# y shape: (10886,)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=333)



scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# y_test = y_test.reshape(-1,1)
x_train = x_train.to_numpy().reshape(-1, 4,2, 1)
x_test = x_test.to_numpy().reshape(-1, 4,2, 1)
y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)


# 4. Build Model
model = Sequential()

model.add(Conv2D(64, (2,2), input_shape=(4,2, 1), activation='relu', padding='same'))
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


# loss: 22039.21875
# mae: 112.27574920654297
# 걸린시간: 182.80455708503723