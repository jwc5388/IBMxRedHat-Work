#copied from 18-5

import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
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



path_mcp = 'Study25/_save/keras28_mcp/05_kaggle_bike/'
model = load_model(path + 'keras28_kaggle_bike_save.h5')


# Compile with better learn rate
model.compile(loss='mse', optimizer='adam')


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


