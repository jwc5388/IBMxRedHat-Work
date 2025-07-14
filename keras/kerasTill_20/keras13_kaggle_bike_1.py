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

# CRITICAL FIX: Apply scaling to both training and test data
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
test_csv_scaled = scaler.transform(test_csv)

# Build improved model with correct input dimension
model = Sequential()
model.add(Dense(128, input_dim=8, activation='relu'))  # input_dim matches your 8 features
model.add(Dense(64, activation='relu'))   # Reduced complexity to prevent overfitting
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# Compile with better learning rate
model.compile(loss='mse', optimizer='adam')

# Train with validation data and more reasonable epochs
history = model.fit(x_train_scaled, y_train, 
                   epochs=50,  # Reduced from 100 to prevent overfitting
                   batch_size=32,  # Increased batch size for stability
                   validation_data=(x_test_scaled, y_test),
                   verbose=1)

# Evaluate and predict
loss = model.evaluate(x_test_scaled, y_test)
print("loss:", loss)

result = model.predict(x_test_scaled)
r2 = r2_score(y_test, result)
print("r2 score:", r2)

rmse = np.sqrt(mean_squared_error(y_test, result))
print("RMSE:", rmse)

# Make predictions on test set using scaled data
y_submit = model.predict(test_csv_scaled)
print(y_submit.shape)

submission_csv['count'] = y_submit
print(submission_csv.head())
submission_csv.to_csv(path_save + 'submission_0522_0103.csv')