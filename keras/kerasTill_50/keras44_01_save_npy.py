import numpy as np
import os
import time
from keras.preprocessing.image import ImageDataGenerator

# 경로 설정
if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    basepath = '/workspace/TensorJae/Study25/'
else:
    basepath = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')

path_train = basepath + '_data/kaggle/cat_dog/train2/'
path_test = basepath + '_data/kaggle/cat_dog/test2/'
np_path = basepath + '_save/save_npy/'

# ImageDataGenerator 생성
train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

# 트레인 데이터 로드
xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(150, 150),
    batch_size=100,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
    seed=42
)

# 테스트 데이터 로드
xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(150, 150),
    batch_size=100,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False
)

### 모든 배치 데이터를 하나로 합치기 (train)
all_x_train = []
all_y_train = []

for i in range(len(xy_train)):
    x_batch, y_batch = xy_train[i]
    all_x_train.append(x_batch)
    all_y_train.append(y_batch)
    if (i+1)*xy_train.batch_size >= xy_train.samples:
        break

x_train = np.concatenate(all_x_train, axis=0)
y_train = np.concatenate(all_y_train, axis=0)

### 모든 배치 데이터를 하나로 합치기 (test)
all_x_test = []
all_y_test = []

for i in range(len(xy_test)):
    x_batch, y_batch = xy_test[i]
    all_x_test.append(x_batch)
    all_y_test.append(y_batch)
    if (i+1)*xy_test.batch_size >= xy_test.samples:
        break

x_test = np.concatenate(all_x_test, axis=0)
y_test = np.concatenate(all_y_test, axis=0)

### .npy 파일로 저장
os.makedirs(np_path, exist_ok=True)

np.save(np_path + 'keras44cd_x_train.npy', x_train)
np.save(np_path + 'keras44cd_y_train.npy', y_train)
np.save(np_path + 'keras44cd_x_test.npy', x_test)
np.save(np_path + 'keras44cd_y_test.npy', y_test)

### 확인
print('✅ 저장 완료:')
print('x_train:', x_train.shape)
print('y_train:', y_train.shape)
print('x_test :', x_test.shape)
print('y_test :', y_test.shape)