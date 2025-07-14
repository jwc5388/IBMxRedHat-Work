from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time


"""import ssl
ssl.create_default_https_context = ssl._create_unverified_context
"""


datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)
print(np.unique(y, return_counts = True))
print(pd.value_counts(y))
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


y = pd.get_dummies(y).values

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, train_size= 0.8, random_state=42, stratify= y)

print(np.unique(y_train, return_counts = True))
print(np.unique(y_test, return_counts = True))


model = Sequential()

model.add(Dense(128, input_dim = 54, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(7, activation='softmax'))

model.compile(
    loss = 'categorical_crossentropy',
    optimizer= 'adam',
    metrics = ['acc']
)


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

# cpu = 67.11176180839539

# gpu = 