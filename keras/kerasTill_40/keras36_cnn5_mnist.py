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

# x 를 reshape -> (60000,28,28,1)
x_train = x_train.reshape(60000,28,28,1)
# x_test = x_test.reshape(10000,28,28,1)
#위아래 똑같음 잘 봐
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)
print(x_train.shape, x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)  #(60000, 10) (10000, 10)


#2 model 
model = Sequential([
    Conv2D(filters=64,kernel_size= (3,3), strides = 1, input_shape = (28,28,1), activation = 'relu'),
    BatchNormalization(),
    Conv2D(filters = 64, kernel_size=(3,3)),
    BatchNormalization(),
    Dropout(0.2),
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
y_test = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_test, y_pred)

#colab gpu 
# loss: 2.012453317642212
# acc: 0.20559999346733093
# 걸린시간: 109.68927764892578

#server gpu
# loss: 0.11768920719623566
# acc: 0.9677000045776367
# 걸린시간: 433.2018299102783

#server gpu layer added
# loss: 0.20301711559295654
# acc: 0.9711999893188477
# 걸린시간: 1170.1360108852386

#cpu
# loss: 2.3011512756347656
# acc: 0.11349999904632568
# 걸린시간: 3850.199594974518

