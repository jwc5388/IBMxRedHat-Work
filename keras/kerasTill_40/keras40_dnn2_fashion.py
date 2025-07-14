import numpy as np
import pandas as pd
import time
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical

# 1. Load Data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape)  # (60000, 28, 28), (60000,)
print(x_test.shape, y_test.shape)    # (10000, 28, 28), (10000,)


x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print(y_train.shape, y_test.shape)

# exit()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model = Sequential()

model.add(Dense(128, input_shape =(28*28,), activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())


model.add(Dense(10, activation = 'softmax'))

#3 compile 

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode = 'auto',
    patience=10,
    restore_best_weights=True
)


start = time.time()

hist = model.fit(x_train, y_train, epochs= 200, batch_size=64,validation_split =0.2, verbose=1,callbacks=[es])

end = time.time()



loss, acc = model.evaluate(x_test, y_test, verbose=1)

print('loss:', loss)
print('acc:', acc)

print('걸린시간:', end-start)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_test, y_pred)


# loss: 0.4034295976161957
# acc: 0.8859999775886536
# 걸린시간: 702.7561140060425