
#copied from 18-5

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time

path = 'Study25/_data/kaggle/bike/'
path_save = 'Study25/_data/kaggle/bike/csv_files/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path+ 'sampleSubmission.csv', index_col=0)


# Prepare features and target
x = train_csv.drop(['casual', 'registered', 'count'], axis = 1)
y = train_csv['count']

print(f"X shape: {x.shape}")
print(f"y shape: {y.shape}")

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=333)


# Build improved model with correct input dimension
model = Sequential()
model.add(Dense(128, input_dim=8, activation='relu'))  # input_dim matches your 8 features
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))   # Reduced complexity to prevent overfitting
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))  
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))  
model.add(Dense(1))



# Compile with better learn rate
model.compile(loss='mse', optimizer='adam', metrics = ['acc'])



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


#With dropout  
# loss: 24858.26953125
# accuracy: 0.010286553762853146
# cpu = 36.014272928237915

# gpu = 