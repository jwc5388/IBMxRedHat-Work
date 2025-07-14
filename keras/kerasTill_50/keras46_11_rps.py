import numpy as np
import pandas as pd
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
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
# y = to_categorical(y, num_classes=3)

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=42)
print(f"데이터 로딩 완료. 소요시간: {round(time.time() - start, 2)}초")
print(x_train.shape, y_train.shape)

# ✅ 모델 구성
model = Sequential()
model.add(Conv2D(128, (3,3), padding='same', activation='relu', input_shape=(150,150,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))  # ✅ 다중 클래스 분류

# ✅ 손실함수, 옵티마이저, 콜백
loss_fn = CategoricalCrossentropy(label_smoothing=0.02)
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1)

mcp_path = '/workspace/TensorJae/Study25/_save/keras46_rps/'

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,
    filepath = mcp_path + 'keras46_rps.h5'
)
# ✅ 모델 학습
start = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[es, lr],
    verbose=1
)
print(f"모델 학습 완료. 소요시간: {round(time.time() - start, 2)}초")

# ✅ 평가
loss, acc = model.evaluate(x_train, y_train, verbose=0)
print(f"최종 훈련 데이터 평가 - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# 모델 학습 완료. 소요시간: 385.58초
# 최종 훈련 데이터 평가 - Loss: 0.0875, Accuracy: 0.9963