import numpy as np
import pandas as pd
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, Dropout, Flatten, Dense, MaxPooling1D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from datetime import datetime

# ✅ 경로 설정
np_path = '/workspace/TensorJae/Study25/_save/save_npy/'
sample_path = '/workspace/TensorJae/Study25/_data/kaggle/cat_dog/sample_submission.csv'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# ✅ 데이터 불러오기 + 정규화
start = time.time()
x = np.load(np_path + 'save_npykeras44_x_train_new.npy')
y = np.load(np_path + 'save_npykeras44_y_train_new.npy')
test = np.load(np_path + 'save_npykeras44_x_test.npy')

print(x.shape)  # (25000, 150, 150, 3)
print(y.shape)  # (25000,)

augment_size = 5000
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=0.1
)

randidx = np.random.randint(x.shape[0], size=augment_size)
x_augmented = x[randidx].copy()
y_augmented = y[randidx].copy()

x_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False
).next()[0]

x = np.concatenate((x, x_augmented))
y = np.concatenate((y, y_augmented))

print(f"✅ 증강 후: {x.shape}, {y.shape}")  # (30000, 150, 150, 3)

# ✅ reshape to Conv1D input (samples, timesteps, features)
x = x.reshape(x.shape[0], -1, 3)        # (30000, 150*150, 3) = (30000, 22500, 3)
test = test.reshape(test.shape[0], -1, 3)  # (12500, 22500, 3)

# ✅ 훈련/검증 분할
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=42)
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")  # e.g. (24000, 22500, 3)

# ✅ Conv1D 모델 구성
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=(22500, 3)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.2),

    Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.2),

    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# ✅ 컴파일
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=BinaryCrossentropy(label_smoothing=0.02),
    metrics=['acc']
)

es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1)

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
print(f"모델 학습 완료. ⏱️ 소요시간: {round(time.time() - start, 2)}초")

# ✅ 평가
loss, acc = model.evaluate(x_val, y_val, verbose=0)
print(f"평가 결과 - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# ✅ 예측
pred = model.predict(test, verbose=1).reshape(-1)

# # ✅ 제출 파일 저장
# submission = pd.read_csv(sample_path)
# submission['label'] = pred
# submission.to_csv(f'/workspace/TensorJae/Study25/_save/submission_{timestamp}.csv', index=False)
# print(f"제출 완료 ✅")


# 평가 결과 - Loss: 0.6032, Accuracy: 0.6923