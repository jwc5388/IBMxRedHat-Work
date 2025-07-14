import numpy as np
import pandas as pd
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from datetime import datetime

# ✅ 경로 설정
np_path = '/workspace/TensorJae/Study25/_save/keras46_horses/'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# ✅ 데이터 불러오기
start = time.time()
x = np.load(np_path + 'keras46_horse_x_train.npy') 
y = np.load(np_path + 'keras46_horse_y_train.npy')
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
print(f"데이터 로딩 완료. 소요시간: {round(time.time() - start, 2)}초")
print(f"x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")  # 예: x_train.shape: (821, 150, 150, 3), y_train.shape: (821,)

# ✅ Conv1D 입력용 reshape: (samples, timesteps, features)
x_train = x_train.reshape(x_train.shape[0], -1, 3)  
x_test = x_test.reshape(x_test.shape[0], -1, 3)     

# ✅ Conv1D 모델 구성
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=(150*150, 3)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.2),

    Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.2),

    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.3),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# ✅ 손실함수, 옵티마이저, 콜백
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=BinaryCrossentropy(label_smoothing=0.02),
    metrics=['acc']
)

es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1)

mcp_path = '/workspace/TensorJae/Study25/_save/keras46_horses/'
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=mcp_path + 'keras46_horse.h5'
)

# ✅ 모델 학습
start = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=1,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[es, lr, mcp],
    verbose=1
)
print(f"모델 학습 완료. 소요시간: {round(time.time() - start, 2)}초")

# ✅ 평가
loss, acc = model.evaluate(x_train, y_train, verbose=0)
print(f"최종 훈련 데이터 평가 - Loss: {loss:.4f}, Accuracy: {acc:.4f}")


# 최종 훈련 데이터 평가 - Loss: 0.9299, Accuracy: 0.4896