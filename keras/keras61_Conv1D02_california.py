import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten
from keras.callbacks import EarlyStopping
import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. 데이터 로딩
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333)

# 2. 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 3. Reshape for LSTM
x_train = x_train.reshape(-1, 8, 1)
x_test = x_test.reshape(-1, 8, 1)


# 4. 모델 구성
model = Sequential()
model.add(Conv1D(filters=64, input_shape = (8,1), kernel_size=2, padding='same', activation='relu'))
model.add(Conv1D(filters = 64, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# 5. 컴파일 및 학습
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)


start = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_split=0.2, verbose=1, callbacks= [es])
end = time.time()

# 6. 평가
loss, mae = model.evaluate(x_test, y_test)
result = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, result))
r2 = r2_score(y_test, result)

# 7. 출력
print("\n📊 Evaluation Metrics:")
print(f"✅ loss (MSE): {loss:.4f}")
print(f"✅ MAE: {mae:.4f}")
print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ R² Score: {r2:.4f}")
print(f"⏱️ 걸린시간: {end - start:.2f}초")


# 📊 Evaluation Metrics:
# ✅ loss (MSE): 0.3608
# ✅ MAE: 0.4131
# ✅ RMSE: 0.6007
# ✅ R² Score: 0.7168
# ⏱️ 걸린시간: 815.86초
