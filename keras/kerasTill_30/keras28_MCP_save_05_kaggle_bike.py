#copied from 18-5

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint

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
model.add(Dense(64, activation='relu'))   # Reduced complexity to prevent overfitting
model.add(Dense(64, activation='relu'))  
model.add(Dense(64, activation='relu'))  
model.add(Dense(1))



path_mcp = 'Study25/_save/keras28_mcp/05_kaggle_bike/'
model.save(path_mcp + 'keras28_kaggle_bike_save.h5')


# Compile with better learn rate
model.compile(loss='mse', optimizer='adam')

from keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights = True,
)


import datetime 
date = datetime.datetime.now()
print(date)  #2025-06-02 13:00:52.507403
print(type(date))   #<class 'datetime.datetime'>
#시간을 string으로 만들어라
date = date.strftime("%m%d_%H%M")
print(date) #0602_1305
print(type(date))   # <class 'str'>

filename = '{epoch:04d}-{val_loss:.4f}.h5'
filepath = "".join([path_mcp, 'k28_', date, '_', filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,
    filepath = filepath
)

# Train with validation data and more reasonable epochs
hist = model.fit(x_train, y_train, 
                   epochs=500,  
                   batch_size=32,  # Increased batch size for stability
                   validation_split = 0.2,
                   verbose=1,
                   callbacks = [es,mcp])



# Evaluate and predict
loss = model.evaluate(x_test, y_test)
print("loss:", loss)

result = model.predict(x_test)
r2 = r2_score(y_test, result)
print("r2 score:", r2)

rmse = np.sqrt(mean_squared_error(y_test, result))
print("RMSE:", rmse)

# Make predictions on test set using scaled data
y_submit = model.predict(test_csv)
print(y_submit.shape)

submission_csv['count'] = y_submit
print(submission_csv.head())
submission_csv.to_csv(path_save + 'submission_0526_13xx.csv')


