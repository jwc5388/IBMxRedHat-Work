
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint

import time


# 1. Load Data
path = 'Study25/_data/kaggle/bank/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. Encode categorical features
le_geo = LabelEncoder()
le_gender = LabelEncoder()

train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])

test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

# 3. Drop unneeded columns
train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

# 4. Separate features and target
x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

print(x.shape)
print(y.shape)


# (165034, 10)
# (165034,)

# exit()


# 6. Train-test split (after scaling)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=33
)
# 5. Apply MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# y_test = y_test.reshape(-1,1)
x_train = x_train.to_numpy().reshape(-1, 5,2, 1)
x_test = x_test.to_numpy().reshape(-1, 5,2, 1)
y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)


# 4. Build Model
model = Sequential()

model.add(Conv2D(64, (2,2), input_shape=(5,2, 1), activation='relu', padding='same'))
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
    patience=10,              # 10 epoch 개선 없으면 멈춤
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