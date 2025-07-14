from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
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

#   print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747
# Name: count, dtype: int64

# (581012, 54) (581012,)
# (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))



y = pd.get_dummies(y).values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state=42, stratify= y)



# scaler = MinMaxScaler()


# scaler = StandardScaler()


# scaler = MaxAbsScaler()


scaler = RobustScaler()
# loss: 0.27138957381248474
# accuracy: 0.8916292786598206
# r2 score: 0.7340049743652344

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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


es = EarlyStopping(
    monitor= 'val_loss',
    mode='min',
    patience= 20,
    restore_best_weights= True
)



model.fit(
    x_train, y_train, 
    epochs=500, 
    batch_size=256, 
    callbacks=[es], 
    validation_split= 0.2, 
    verbose =1)



loss, accuracy = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accuracy:', accuracy)

result = model.predict(x_test)
r2 = r2_score(y_test, result)
print('r2 score:', r2)


