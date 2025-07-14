
#copied from 18-5

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Flatten, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time

path = '/workspace/TensorJae/Study25/_data/kaggle/bike/'
path_save = '/workspace/TensorJae/Study25/_data/kaggle/bike/csv_files/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path+ 'sampleSubmission.csv', index_col=0)


# Prepare features and target
x = train_csv.drop(['casual', 'registered', 'count'], axis = 1)
y = train_csv['count']

print(f"X shape: {x.shape}")
print(f"y shape: {y.shape}")

# X shape: (10886, 8)
# y shape: (10886,)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=333)



scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


x_train = x_train.reshape(-1, 8, 1)
x_test = x_test.reshape(-1, 8, 1)

model = Sequential([
    Conv1D(filters = 64, kernel_size = 2, padding = 'same', input_shape = (8,1), activation = 'relu'),
    Conv1D(filters = 64, kernel_size = 2, activation= 'relu'),
    Flatten(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1) 
])
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 20,
    restore_best_weights = True,
)
# === Train ===
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

# === Evaluate ===
loss, mae = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test)

# === R2 & RMSE ===
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# === Print results ===
print(f'✅ Loss (MSE): {loss:.4f}')
print(f'✅ MAE        : {mae:.4f}')
print(f'✅ RMSE       : {rmse:.4f}')
print(f'✅ R2 Score   : {r2:.4f}')
print(f'✅ Training Time: {end - start:.2f} seconds')


# ✅ Loss (MSE): 21617.4414
# ✅ MAE        : 108.1142
# ✅ RMSE       : 147.0287
# ✅ R2 Score   : 0.3272
# ✅ Training Time: 117.30 seconds



