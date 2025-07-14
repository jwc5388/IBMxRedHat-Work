
#https://dacon.io/competitions/open/235576/overview/description

import numpy as np      #1.23.0
import pandas as pd     #2.2.3

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Flatten, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time 
#1 data
# train_csv = pd.read_csv('.\_data\dacon\따릉이\train.csv')
## get rid of index column and just get the data

path = '/workspace/TensorJae/Study25/_data/dacon/ddarung/'
path_save = '/workspace/TensorJae/Study25/_data/dacon/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
# print(train_csv) #[1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
# print(test_csv) #[715 rows x 9 columns]

submission_csv = pd.read_csv(path+ 'submission.csv', index_col=0)
# print(submission_csv) # [715 rows x 1 columns]

train_csv = train_csv.dropna()


# test_csv = test_csv.dropna()
test_csv = test_csv.fillna(train_csv.mean())
print(test_csv.info())

x = train_csv.drop(['count'], axis=1)  #count라는 axis=1 열 삭제, 행은 axis =0
print(x) #[1459 rows x 9 columns]
# y = train_csv.

#take only the count colum to y 
y = train_csv['count'] 
print(y) #(1459,)



x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state= 3333)

x_train = x_train.to_numpy().reshape(-1, 9, 1)
x_test = x_test.to_numpy().reshape(-1, 9, 1)


model = Sequential([
    Conv1D(filters = 64, kernel_size = 2, padding = 'same', input_shape = (9,1), activation = 'relu'),
    Conv1D(filters = 64, kernel_size = 2, activation= 'relu'),
    Flatten(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1) 
])

model.compile(loss='mse', optimizer='adam', metrics=['mae']) 

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    restore_best_weights=True,
    patience = 5,
)

start = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    verbose=1,
    validation_split=0.2,
    callbacks = [es]
)
end = time.time()

loss, mae = model.evaluate(x_test, y_test)
result = model.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, result))
r2 = r2_score(y_test, result)


print("\n📊 Final Evaluation:")
print(f"✅ Loss (MSE): {loss:.4f}")
print(f"✅ MAE       : {mae:.4f}")
print(f"✅ RMSE      : {rmse:.4f}")
print(f"✅ R² Score  : {r2:.4f}")
print(f"⏱️ 걸린시간   : {end - start:.2f}초")


# 📊 Final Evaluation:
# ✅ Loss (MSE): 2169.1875
# ✅ MAE       : 31.3066
# ✅ RMSE      : 46.5745
# ✅ R² Score  : 0.6889
# ⏱️ 걸린시간   : 78.24초