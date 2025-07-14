import numpy as np
import time
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os

# ✅ 경로 설정
path_train = '/Users/jaewoo000/Desktop/IBM:RedHat/Study25/_data/tensor_cert/rps/'
np_path = '/Users/jaewoo000/Desktop/IBM:RedHat/Study25/_save/rps/'

# ✅ ImageDataGenerator (스케일링만 적용)
train_datagen = ImageDataGenerator(rescale=1./255)

# ✅ 이미지 로딩
xy_data = train_datagen.flow_from_directory(
    path_train,
    target_size=(150, 150),
    batch_size=100,  # 전체 데이터를 가져올 수 있게 충분히 크게
    class_mode='categorical',  # ✅ 다중 분류니까 categorical
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
        break

x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)
print('전체 데이터:', x.shape, y.shape)  # 예: (2520, 150, 150, 3) (2520, 3)

# ✅ train/test 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=42
)

print('x_train:', x_train.shape, 'y_train:', y_train.shape)
print('x_test:', x_test.shape, 'y_test:', y_test.shape)

# ✅ 저장
os.makedirs(np_path, exist_ok=True)
np.save(np_path + 'keras_rps_x_train.npy', x_train)
np.save(np_path + 'keras_rps_y_train.npy', y_train)
np.save(np_path + 'keras_rps_x_test.npy', x_test)
np.save(np_path + 'keras_rps_y_test.npy', y_test)

print("✅ 저장 완료")