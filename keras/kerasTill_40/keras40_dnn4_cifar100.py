from keras.datasets import cifar100
import numpy as np
import pandas as pd
import time
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical


# 1. Load data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)   
# (50000, 32, 32, 3) (50000, 1)
# (10000, 32, 32, 3) (10000, 1)

# print
# x_train = x_train / 255.
# x_test = x_test / 255.
# y_train = pd.get_dummies(y_train.reshape(-1))
# y_test = pd.get_dummies(y_test.reshape(-1))

x_train = x_train.reshape(-1,32*32*3)
x_test = x_test.reshape(-1, 32*32*3)

print(x_train.shape)
print(x_test.shape)

# exit()


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


print(y_train.shape, y_test.shape) #(50000, 10) (10000, 10)

exit()

model = Sequential()

model.add(Dense(128, input_shape =(32*32*3,), activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())


model.add(Dense(100), activation = 'softmax')


#3 compile 

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode = 'auto',
    patience=10,
    restore_best_weights=True
)


start = time.time()

hist = model.fit(x_train, y_train, epochs= 500, batch_size=64, validation_split = 0.2, verbose=1,callbacks=[es])

end = time.time()



loss, acc = model.evaluate(x_test, y_test)

print('loss:', loss)
print('acc:', acc)

print('걸린시간:', end-start)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_test, y_pred)

print('accurcy score:', acc_score)

# loss: 3.2478747367858887
# acc: 0.22660000622272491
# 걸린시간: 947.0528604984283