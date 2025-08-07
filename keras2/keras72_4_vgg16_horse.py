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
# from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import random
import os

from keras.applications import VGG16
SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

# ✅ 경로 설정
np_path = basepath + '_save/horse/'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# ✅ 데이터 불러오기
start = time.time()
x = np.load(np_path + 'keras46_03_x_train.npy') 
y = np.load(np_path + 'keras46_03_y_train.npy')
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
print(f"데이터 로딩 완료. 소요시간: {round(time.time() - start, 2)}초")
print(f"x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")  # 예: x_train.shape: (821, 150, 150, 3), y_train.shape: (821,)


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
model.add(Dense(2, activation='softmax'))

model.summary()


model.compile(loss= 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True,verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

start = time.time()
model.fit(x_train, y_train,
          epochs=100,
          batch_size=32,
          validation_split=0.2,
          callbacks=[es,lr],
          verbose=1)
end = time.time()

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('evaluationnnnnnnnnnn')
print(f'loss: {loss:.4f}')
print(f'accuracy: {acc:.4f}')
print(f'Time: {end - start:.2f}seconds')

y_pred= model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
# y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print(f'accuracy score: {acc_score}:.4f')



# 가중치 동결 했을때,

### 실습 ###
#비교할거
# 1. 이번의 본인이 한 최상의 결과가
# 2. 가중치를 동결하지 않고 훈련시켰을때, trainable = True
# 3. 가중치를 동결하고 훈련시켰을때, trainable = False
# 시간까지 비교하기





###추가###
# Flatten 과 GAP



###
# cifar10
# cifar100
# horse

