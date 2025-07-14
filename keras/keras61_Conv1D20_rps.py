import numpy as np
import pandas as pd
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from datetime import datetime

# ✅ 경로 설정
np_path = '/workspace/TensorJae/Study25/_save/keras46_rps/'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# ✅ 데이터 불러오기 + 정규화
start = time.time()
x = np.load(np_path + 'keras46_rps_x_train.npy') / 255.0
y = np.load(np_path + 'keras46_rps_y_train.npy')

# ✅ 레이블 원-핫 인코딩
y = to_categorical(y, num_classes=3)

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=42)
print(f"데이터 로딩 완료. 소요시간: {round(time.time() - start, 2)}초")
print(f"x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")

# ✅ Conv1D 입력용으로 reshape (N, 타임스텝, 피처)
x_train = x_train.reshape(x_train.shape[0], -1, 3)  # (N, 22500, 3)
x_val = x_val.reshape(x_val.shape[0], -1, 3)

# ✅ Conv1D 모델 구성
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(x_train.shape[1], 3)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(256, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

# ✅ 손실함수, 옵티마이저, 콜백
loss_fn = CategoricalCrossentropy(label_smoothing=0.02)
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1)

mcp_path = '/workspace/TensorJae/Study25/_save/keras46_rps/'
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=mcp_path + 'keras46_rps.h5'
)

# ✅ 모델 학습
start = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[es, lr, mcp],
    verbose=1
)
print(f"모델 학습 완료. 소요시간: {round(time.time() - start, 2)}초")

# ✅ 평가
loss, acc = model.evaluate(x_train, y_train, verbose=0)
print(f"최종 훈련 데이터 평가 - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
