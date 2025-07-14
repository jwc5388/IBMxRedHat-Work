import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

# Load Data
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

x = data
y = target
print(x.shape, y.shape)  # (506, 13) (506,)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

# Feature scaling (선택적으로 사용 가능)
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# Reshape for LSTM
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)  # (N, 13, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
print("x_train shape:", x_train.shape)

# Define LSTM Model
model = Sequential([
    LSTM(64, input_shape=(13, 1)),       # (N, 13, 1)
    Dense(16, activation='relu'),        # (N, 16)
    Dropout(0.2),
    Dense(1)                             # (N, 1)
])

model.summary()

# Callbacks
path = 'Study25/_save/keras28_mcp/01_boston/'
model.save(path + 'keras28_boston_save.h5')

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=5,
    restore_best_weights=True
)

# 파일 이름 설정
date = datetime.datetime.now().strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.h5'
filepath = "".join([path, 'k28_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=filepath
)

# Compile & Train
model.compile(loss='mse', optimizer='adam')

hist = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[es, mcp]
)

# Evaluation
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print Results
print(f"✅ Final Evaluation:")
print(f"Loss (MSE): {loss:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


# ✅ Final Evaluation:
# Loss (MSE): 484.6977
# RMSE: 22.0159
# R² Score: -5.6095