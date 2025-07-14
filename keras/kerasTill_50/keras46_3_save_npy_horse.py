import numpy as np
import time
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

# ✅ 경로 설정
path_train = basepath + '_data/tensor_cert/horse-or-human/'
np_path = basepath + '_save/horse/'  # 저장 경로

# ✅ 이미지 증강 및 정규화 (스케일링만 적용)
train_datagen = ImageDataGenerator(rescale=1./255)

# ✅ 데이터 로딩
xy_data = train_datagen.flow_from_directory(
    path_train,
    target_size=(150, 150),
    batch_size=100,  # 모든 이미지를 불러오려면 충분히 큰 배치 사이즈
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
    seed=42
)

# ✅ 모든 배치 합치기
all_x = []
all_y = []

for i in range(len(xy_data)):
    x_batch, y_batch = xy_data[i]
    all_x.append(x_batch)
    all_y.append(y_batch)
    if len(all_x) * xy_data.batch_size >= xy_data.samples:
        break  # 모든 데이터 로딩했으면 종료

x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)
print('전체 데이터:', x.shape, y.shape)

# ✅ train/test 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=42)

print('train:', x_train.shape, y_train.shape)
print('test:', x_test.shape, y_test.shape)

# ✅ 저장
os.makedirs(np_path, exist_ok=True)

np.save(np_path + 'keras_horse_x_train.npy', x_train)
np.save(np_path + 'keras_horse_y_train.npy', y_train)
np.save(np_path + 'keras_horse_x_test.npy', x_test)
np.save(np_path + 'keras_horse_y_test.npy', y_test)

print("✅ 저장 완료")