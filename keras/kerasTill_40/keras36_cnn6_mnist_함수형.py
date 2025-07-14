import numpy as np
import pandas as pd

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, Dropout, Flatten, BatchNormalization

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import time

from sklearn.metrics import accuracy_score


#CNN
#1 data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)



# ##########scaling 1. MinMaxScaler()##############
#여기는 minmaxscaler 를 하기 위해 reshaping 을 하는 것이다.
# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]* x_test.shape[2])
# print(x_train.shape, x_test.shape)  #(60000, 784) (10000, 784)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

# print(x_train.shape, x_test.shape)
# print(np.max(x_train), np.min(x_train)) #1.0 0.0

# print(np.max(x_test), np.min(x_test)) #24.0 0.0

############# scaling 2. #####################
# x_train = x_train/255.
# x_test = x_test/255.
# print(np.max(x_train), np.min(x_train)) #1.0 0.0

# print(np.max(x_test), np.min(x_test)) #1.0 0.0

#3 스케일링(많이쓴다.)
x_train = (x_train - 127.5)/127.5
x_test = (x_test-127.5)/127.5
print(np.max(x_train), np.min(x_train))  #1.0 -1.0
print(np.max(x_test), np.min(x_test))   #1.0 -1.0


y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)  #(60000, 10) (10000, 10)


# #2 model 
# model = Sequential([
#     Conv2D(filters=64,kernel_size= (3,3), strides = 1, input_shape = (28,28,1), activation = 'relu'),
#     BatchNormalization(),
#     Conv2D(filters = 64, kernel_size=(3,3)),
#     Dropout(0.2),
#     BatchNormalization(),
#     Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
#     Flatten(),
#     Dense(units=16, activation='relu'),
#     Dropout(0.2),
#     Dense(units=16, input_shape=(16,)),
#     Dense(units=10, activation='softmax'),
    
# ])


input1 = Input(shape = (28,28,1))
conv1 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(input1)
batch1 = BatchNormalization()(conv1)
conv2 = Conv2D(filters=64, kernel_size=(3,3))(batch1)
drop1 = Dropout(0.2)(conv2)
batch2 = BatchNormalization()(drop1)
conv3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu')(batch2)
flat = Flatten()(conv3)
dense1 = Dense(units=16, activation='relu')(flat)
drop2 = Dropout(0.2)(dense1)
dense2 = Dense(units=16, input_shape=(16,))(drop2)
output1 = Dense(units=10, activation='softmax')(dense2)

model = Model(inputs = input1, outputs = output1)

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
y_test = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_test, y_pred)


