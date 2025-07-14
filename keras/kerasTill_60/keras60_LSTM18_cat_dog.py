import numpy as np
import pandas as pd
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from datetime import datetime

# ✅ 경로 설정
np_path = '/workspace/TensorJae/Study25/_save/save_npy/'
path_test = '/workspace/TensorJae/Study25/_data/kaggle/cat_dog/test2/'
sample_path = '/workspace/TensorJae/Study25/_data/kaggle/cat_dog/sample_submission.csv'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# save_path = f'/workspace/TensorJae/Study25/_save/submission_{timestamp}.csv'

# ✅ 데이터 불러오기 + 정규화
start = time.time()
x = np.load(np_path + 'save_npykeras44_x_train_new.npy') / 255.
y = np.load(np_path + 'save_npykeras44_y_train_new.npy')
test = np.load(np_path + 'save_npykeras44_x_test.npy') / 255.
print(x.shape, y.shape)  # (25000, 150, 150, 3), (25000,)
print(f"데이터 로딩 완료 ⏱️ {round(time.time() - start, 2)}초")

# ✅ 데이터 증강
augment_size = 5000
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=0.1
)

randidx = np.random.randint(x.shape[0], size=augment_size)
x_augmented = datagen.flow(x[randidx], y[randidx], batch_size=augment_size, shuffle=False).next()[0]
y_augmented = y[randidx]

x = np.concatenate((x, x_augmented))
y = np.concatenate((y, y_augmented))
print(f"✅ 증강 후: {x.shape}, {y.shape}")  # (30000, 150, 150, 3)

# ✅ Train/Validation 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")

# ✅ LSTM용 reshape
x_train = x_train.reshape(-1, 150*150*3, 1)
x_test = x_test.reshape(-1, 150*150*3, 1)
test = test.reshape(-1, 150*150*3, 1)

# ✅ 모델 구성
model = Sequential([
    LSTM(64, return_sequences=True, dropout=0.2, input_shape=(150*150*3, 1)),
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

# ✅ 학습
start = time.time()
model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[es, lr],
    verbose=1
)
print(f"✅ 학습 완료 ⏱️ {round(time.time() - start, 2)}초")

# ✅ 평가
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"✅ Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

# ✅ 예측
pred = model.predict(test, verbose=1)
pred_prob = pred.reshape(-1)

# ✅ 제출 저장 (주석 해제 시)
# submission = pd.read_csv(sample_path)
# submission['label'] = pred_prob
# submission.to_csv(save_path, index=False)
# print(f"✅ 제출 파일 저장 완료: {save_path}")
