import sklearn as sk
import tensorflow as tf
import numpy as np
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from keras.optimizers import Adam, Adagrad, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import r2_score, accuracy_score


# 1. 데이터 준비
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333)


model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(100))
model.add(Dense(100, input_dim=8))

# 2. 탐색할 옵티마이저 및 러닝레이트
optimizers = [Adam, Adagrad, SGD]
lr_list = [0.1, 0.01, 0.05, 0.001, 0.0005, 0.0001]


model = Sequential([
Dense(128, input_dim=8, activation='relu'),
Dense(128, activation='relu'),
Dense(64, activation='relu'),
Dense(1)
])

model.compile(loss = 'mse', optimizer=Adam(learning_rate=0.01))

es = EarlyStopping(monitor = 'val_loss',
                   mode = 'min',
                   patience = 20,
                   verbose = 1,
                   restore_best_weights=True)

rlr = ReduceLROnPlateau(monitor = 'val_loss', mode = 'auto', patience = 10,
                        verbose = 1,
                        factor = 0.9)

# 0.1 / 0.05/ 0.025/ 0.0125/ 0.00625 ##factor 0.5

start = time.time()
hist = model.fit(x_train, y_train, epochs = 10000, batch_size =32, verbose = 1, validation_split=0.1,
                 callbacks = [es, rlr])




loss, acc = model.evaluate(x_test, y_test)
result = model.predict(x_test)

acc = accuracy_score(y_test, result)
print("loss:", loss)
print('acc result:', acc)

