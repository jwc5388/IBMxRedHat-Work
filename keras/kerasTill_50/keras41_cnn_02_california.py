import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D, BatchNormalization
import time

# 데이터 로딩
dataset = fetch_california_housing()
x = dataset.data  # (20640, 8)
y = dataset.target

# # 1. ✅ Zero-padding: 8 → 9 → reshape to (3, 3, 1)
# x_padded = np.pad(x, ((0, 0), (0, 1)), mode='constant')  # shape → (20640, 9)
# x_reshaped = x_padded.reshape(-1, 3, 3, 1)  # shape → (20640, 3, 3, 1)

# # 2. 정규화
# scaler = MinMaxScaler()
# x_scaled = scaler.fit_transform(x_reshaped.reshape(len(x_reshaped), -1)).reshape(-1, 3, 3, 1)

# 3. train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1,4,2,1)
x_test = x_test.reshape(-1,4,2,1)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# 4. 모델 구성 (CNN)
model = Sequential()
model.add(Conv2D(32, (2, 2), padding='same', activation='relu', input_shape=(4,2, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))  # 회귀 문제

# 5. 컴파일 및 학습
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
end = time.time()

# 6. 평가
loss, mae = model.evaluate(x_test, y_test)
print(f'✅ loss (MSE): {loss:.4f}')
print(f'✅ mae: {mae:.4f}')
print("⏱️ 걸린시간:", end - start)


# ✅ loss (MSE): 0.3749
# ✅ mae: 0.4276
# ⏱️ 걸린시간: 140.869952917099