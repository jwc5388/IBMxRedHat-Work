#https://dacon.io/competitions/official/236068/overview/description


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Flatten
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

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state= 33)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# y_test = y_test.reshape(-1,1)
x_train = x_train.to_numpy().reshape(-1, 4, 2, 1)
x_test = x_test.to_numpy().reshape(-1, 4, 2, 1)
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


es = EarlyStopping(
    monitor='val_loss',       # 기준: 검증 손실
    patience=15,              # 10 epoch 개선 없으면 멈춤
    mode='min',               # 손실이므로 'min'
    verbose=1,
    restore_best_weights=True
)


start = time.time()

# 2. model.fit()에 callbacks 인자로 추가
hist = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[es]  # 👈 여기에 추가!
)



end = time.time()

loss, mae = model.evaluate(x_test, y_test)
print('loss:', loss)
print('mae:', mae)
print('걸린시간:', end - start)


# loss: 0.1515120267868042
# mae: 0.3116854131221771
# 걸린시간: 13.29417085647583