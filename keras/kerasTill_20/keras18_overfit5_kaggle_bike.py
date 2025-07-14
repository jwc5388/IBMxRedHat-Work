import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

path = 'Study25/_data/kaggle/bike/'
path_save = 'Study25/_data/kaggle/bike/csv_files/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path+ 'sampleSubmission.csv', index_col=0)

print(train_csv.columns)
print(test_csv.columns)

# Prepare features and target
x = train_csv.drop(['casual', 'registered', 'count'], axis = 1)
y = train_csv['count']

print(f"X shape: {x.shape}")
print(f"y shape: {y.shape}")

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333)


# Build improved model with correct input dimension
model = Sequential()
model.add(Dense(128, input_dim=8, activation='relu'))  # input_dim matches your 8 features
model.add(Dense(64, activation='relu'))   # Reduced complexity to prevent overfitting
model.add(Dense(32, activation='relu'))
model.add(Dense(16))
model.add(Dense(1))

# Compile with better learning rate
model.compile(loss='mse', optimizer='adam')

# Train with validation data and more reasonable epochs
hist = model.fit(x_train, y_train, 
                   epochs=300,  # Reduced from 100 to prevent overfitting
                   batch_size=32,  # Increased batch size for stability
                   validation_split = 0.2,
                   verbose=1)

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6)) # 9x6 size 
plt.plot(hist.history['loss'], c = 'red', label = 'loss') # y값만 넣으면 시간순으로 그림 그림
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss')
plt.title("Kaggle Bike loss")
plt.xlabel('epoch')
plt.ylabel('loss') 
plt.legend(loc='upper right') #우측 상단에 label 표시
plt.grid() #격자 표시


plt.show()


# # Evaluate and predict
# loss = model.evaluate(x_test, y_test)
# print("loss:", loss)

# result = model.predict(x_test)
# r2 = r2_score(y_test, result)
# print("r2 score:", r2)

# rmse = np.sqrt(mean_squared_error(y_test, result))
# print("RMSE:", rmse)

# # Make predictions on test set using scaled data
# y_submit = model.predict(test_csv)
# print(y_submit.shape)

# submission_csv['count'] = y_submit
# print(submission_csv.head())
# submission_csv.to_csv(path_save + 'submission_0522_0103.csv')


# r2 score: 0.27061164379119873
# RMSE: 150.98143770427544