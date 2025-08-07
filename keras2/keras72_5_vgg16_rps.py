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
from keras.applications import VGG16
import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)
# ✅ 경로 설정
np_path = basepath + '_save/rps/'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# ✅ 데이터 불러오기 + 정규화
start = time.time()
x = np.load(np_path + 'keras46_05_x_train.npy') 
y = np.load(np_path + 'keras46_05_y_train.npy')


print(x.shape)
print(y.shape)

# (2048, 100, 100, 3)
# (2048, 3)

# exit()

# # ✅ 레이블 원-핫 인코딩
# y = to_categorical(y, num_classes=3)

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=42)
print(f"데이터 로딩 완료. 소요시간: {round(time.time() - start, 2)}초")
print(f"x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")

vgg16 = VGG16(
    include_top=False,
    input_shape=(100,100,3),

)

vgg16.trainable = False         #가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(3, activation='softmax'))
# ✅ 손실함수, 옵티마이저, 콜백
loss_fn = CategoricalCrossentropy(label_smoothing=0.02)
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1)

mcp_path = basepath + '_save/keras46_rps'
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
