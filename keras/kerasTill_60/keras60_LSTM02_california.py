import numpy as np
import tensorflow as tf
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

# 1. 데이터 로딩
dataset = fetch_california_housing()
x = dataset.data       # (20640, 8)
y = dataset.target     # (20640,)

# 2. 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 8, 1)   # (N, 8, 1)
x_test = x_test.reshape(-1, 8, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# 3. 모델 구성 (LSTM only)
model = Sequential([
    LSTM(128, input_shape=(8, 1)),      # (N, 128)
    Dense(64, activation='relu'),       # (N, 64)
    Dropout(0.2),
    Dense(1)                            # (N, 1)
])
model.summary()

# 4. 콜백 구성
date = datetime.datetime.now().strftime("%m%d_%H%M")
path = 'save_model/cali_lstm_'
filename = f"{path}{date}_{{epoch:04d}}-{{val_loss:.4f}}.h5"

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=filename,
    verbose=1
)

# 5. 모델 컴파일 및 학습
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

start = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es, mcp],
    verbose=1
)
end = time.time()

# 6. 평가 및 결과 출력
loss, mae = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Evaluation Metrics:")
print(f"✅ Loss (MSE): {loss:.4f}")
print(f"✅ MAE: {mae:.4f}")
print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ R² Score: {r2:.4f}")
print(f"⏱️ 걸린시간: {end - start:.2f}초")
