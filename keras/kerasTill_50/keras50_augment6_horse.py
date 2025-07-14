import numpy as np
import pandas as pd
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.losses import BinaryCrossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from datetime import datetime


# path_train = '/workspace/TensorJae/Study25/_data/brain/train/'
# path_test = '/workspace/TensorJae/Study25/_data/brain/test'
np_path = '/workspace/TensorJae/Study25/_save/keras46_horses/'
# ✅ 경로 설정
# np_path = '/workspace/TensorJae/Study25/_save/save_npy/'
# path_test = '/workspace/TensorJae/Study25/_data/kaggle/cat_dog/test2/'
# sample_path = '/workspace/TensorJae/Study25/_data/kaggle/cat_dog/sample_submission.csv'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# save_path = f'/workspace/TensorJae/Study25/_save/submission_horse_{timestamp}.csv'

# ✅ 데이터 불러오기 + 정규화
start = time.time()
x = np.load(np_path + 'keras46_horse_x_train.npy') 
y = np.load(np_path + 'keras46_horse_y_train.npy')
# test = np.load(np_path + 'keras_x_test.npy') / 255.0
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
# print(f"데이터 로딩 완료. 소요시간: {round(time.time() - start, 2)}초")
# print(x_train.shape, y_train.shape)

augment_size = 3000

datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=0.1,
    
)

randidx = np.random.randint(x.shape[0], size = augment_size)

x_augmented = x[randidx].copy()
y_augmented = y[randidx].copy()

x_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size= augment_size,
    shuffle = False,
).next()[0]

x = np.concatenate((x, x_augmented))
y = np.concatenate((y, y_augmented))

print(f"✅ 증강 후: {x.shape}, {y.shape}")  

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
print(f"데이터 준비 완료 ⏱️ {round(time.time() - start, 2)}초")
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")


# ✅ 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(150,150,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))  # 🔹 약하게

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))  # 🔹 제일 깊은 층만 살짝 강하게

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# ✅ 손실함수, 옵티마이저, 콜백
loss_fn = BinaryCrossentropy(label_smoothing=0.02)
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1)


mcp_path = '/workspace/TensorJae/Study25/_save/keras46_horses/'
mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,
    filepath = mcp_path + 'keras46_horse.h5'
)


# ✅ 모델 학습
start = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[es, lr, mcp],
    verbose=1
)
print(f"모델 학습 완료. 소요시간: {round(time.time() - start, 2)}초")

# ✅ 평가
loss, acc = model.evaluate(x_train, y_train, verbose=0)
print(f"최종 훈련 데이터 평가 - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# ✅ 예측 및 제출 파일 생성
# pred = model.predict(test, verbose=1)
# pred_prob = pred.reshape(-1)

# submission = pd.read_csv(sample_path)
# submission['label'] = pred_prob
# submission.to_csv(save_path, index=False)
# print(f"✅ 제출 파일 저장 완료: {save_path}")

# 모델 학습 완료. 소요시간: 107.94초
# 최종 훈련 데이터 평가 - Loss: 0.0677, Accuracy: 1.0000

# 모델 학습 완료. 소요시간: 357.23초
# 최종 훈련 데이터 평가 - Loss: 0.0602, Accuracy: 1.0000