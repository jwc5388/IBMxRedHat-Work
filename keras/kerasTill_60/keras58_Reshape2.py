import numpy as np
import pandas as pd

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, Dropout, Flatten, BatchNormalization, Reshape, LSTM

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import time

from sklearn.metrics import accuracy_score

## LSTM 넣어봐!!
#CNN
#1 data

# its shaped like ((train_images, train_labels), (test_images, test_labels))
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

# x_train	Training images (60,000 samples)	(60000, 28, 28)
# y_train	Training labels (0~9)	(60000,)
# x_test	Test images (10,000 samples)	(10000, 28, 28)
# y_test	Test labels	(10000,)

x_train = x_train/255.
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) #1.0 0.0

print(np.max(x_test), np.min(x_test)) #1.0 0.0


y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)  #(60000, 10) (10000, 10)


#2 model 
model = Sequential([
    LSTM(100, input_shape = (28,28)),
    Reshape((10,10,1)),
    # Dense(100, input_shape=(28,28)), #(none, 28, 100)
    # Reshape(target_shape=(28,10,10)), #(N,28,10,10)
    Conv2D(filters=64,kernel_size= (3,3), strides = 1), #input_shape = (28,28,1), activation = 'relu'),   #(N,26,8,64)
    BatchNormalization(),
    Conv2D(filters = 64, kernel_size=(3,3)),
    Dropout(0.2),
    BatchNormalization(),
    Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    Flatten(),
    Dense(units=16, activation='relu'),
    Dropout(0.2),
    Dense(units=16, input_shape=(16,)),
    Dense(units=10, activation='softmax'),
    
])

model.summary()


model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics=['acc'])

from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(
    monitor='val_loss',
    mode = 'auto',
    patience=20,
    verbose=1,
    restore_best_weights=True
)

################세이브 파일명 만들기###################
import datetime 
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
path= 'Study25/_save/keras36_cnn5/'
filename = '{epoch:04d}-{val_loss:.4f}.h5'
filepath = "".join([path, 'k36_', date, '_', filename])
##################################################


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,
    filepath = filepath
)

start = time.time()

#cnn에서는 장수
hist = model.fit(x_train,y_train,epochs=1000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es,mcp])

end = time.time()

loss,acc = model.evaluate(x_test, y_test, verbose=1)

print('loss:', loss)
print('acc:', acc)

print('걸린시간:', end-start)

y_pred = model.predict(x_test)
print(y_pred)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test.values, axis=1)
acc_score = accuracy_score(y_test, y_pred)
print("최종 정확도:", acc_score)

# loss: 0.0566054992377758
# acc: 0.9866999983787537
# 걸린시간: 727.0806784629822