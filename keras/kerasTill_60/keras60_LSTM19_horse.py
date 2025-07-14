import numpy as np
import pandas as pd
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from datetime import datetime

# ✅ 경로 설정
np_path = '/workspace/TensorJae/Study25/_save/keras46_horses/'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# ✅ 데이터 로드 및 전처리
start = time.time()
x = np.load(np_path + 'keras46_horse_x_train.npy') / 255.0  # 정규화
y = np.load(np_path + 'keras46_horse_y_train.npy')
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
print(f"✅ 데이터 로딩 완료 ⏱️ {round(time.time() - start, 2)}초")
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

# ✅ Reshape for LSTM (이미지 → 시계열)
x_train = x_train.reshape(-1, 150 * 150 * 3, 1)
x_test = x_test.reshape(-1, 150 * 150 * 3, 1)

# ✅ 모델 구성
model = Sequential([
    LSTM(64, return_sequences=True, dropout=0.2, input_shape=(150 * 150 * 3, 1)),
    LSTM(64, return_sequences=False, dropout=0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# ✅ 컴파일
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=BinaryCrossentropy(label_smoothing=0.02),
    metrics=['accuracy']
)

# ✅ 콜백
es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1)
mcp = ModelCheckpoint(
    filepath=np_path + 'keras46_horse_best.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# ✅ 학습
start = time.time()
hist = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=200,
    batch_size=32,
    callbacks=[es, lr, mcp],
    verbose=1
)
print(f"✅ 모델 학습 완료 ⏱️ {round(time.time() - start, 2)}초")

# ✅ 평가
loss, acc = model.evaluate(x_train, y_train, verbose=0)
print(f"✅ 훈련 데이터 평가 - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
