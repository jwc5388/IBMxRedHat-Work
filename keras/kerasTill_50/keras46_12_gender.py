# import numpy as np
# import pandas as pd
# import time
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from keras.losses import BinaryCrossentropy
# from keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from datetime import datetime


# # path_train = '/workspace/TensorJae/Study25/_data/brain/train/'
# # path_test = '/workspace/TensorJae/Study25/_data/brain/test'
# np_path = '/workspace/TensorJae/Study25/_save/keras46_men_women/'
# # ✅ 경로 설정
# # np_path = '/workspace/TensorJae/Study25/_save/save_npy/'
# # path_test = '/workspace/TensorJae/Study25/_data/kaggle/cat_dog/test2/'
# # sample_path = '/workspace/TensorJae/Study25/_data/kaggle/cat_dog/sample_submission.csv'
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# # save_path = f'/workspace/TensorJae/Study25/_save/submission_horse_{timestamp}.csv'

# # ✅ 데이터 불러오기 + 정규화
# start = time.time()
# x = np.load(np_path + 'keras46_mw_x_train.npy') / 255.0
# y = np.load(np_path + 'keras46_mw_y_train.npy')
# # test = np.load(np_path + 'keras_x_test.npy') / 255.0
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
# print(f"데이터 로딩 완료. 소요시간: {round(time.time() - start, 2)}초")
# print(x_train.shape, y_train.shape)

# # ✅ 모델 구성
# model = Sequential()
# model.add(Conv2D(128, (3,3), padding='same', activation='relu', input_shape=(150,150,3)))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
# model.add(Dropout(0.2))  # 🔹 약하게

# model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))

# model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))  # 🔹 제일 깊은 층만 살짝 강하게

# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))

# # ✅ 손실함수, 옵티마이저, 콜백
# loss_fn = BinaryCrossentropy(label_smoothing=0.02)
# optimizer = Adam(learning_rate=0.001)

# model.compile(optimizer=optimizer, loss=loss_fn, metrics=['acc'])

# es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
# lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1)

# # ✅ 모델 학습
# start = time.time()
# hist = model.fit(
#     x_train, y_train,
#     epochs=200,
#     batch_size=32,
#     validation_data=(x_test, y_test),
#     callbacks=[es, lr],
#     verbose=1
# )
# print(f"모델 학습 완료. 소요시간: {round(time.time() - start, 2)}초")

# # ✅ 평가
# loss, acc = model.evaluate(x_train, y_train, verbose=0)
# print(f"최종 훈련 데이터 평가 - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# # ✅ 예측 및 제출 파일 생성
# # pred = model.predict(test, verbose=1)
# # pred_prob = pred.reshape(-1)

# # submission = pd.read_csv(sample_path)
# # submission['label'] = pred_prob
# # submission.to_csv(save_path, index=False)
# # print(f"✅ 제출 파일 저장 완료: {save_path}")


# # Epoch 25: early stopping
# # 모델 학습 완료. 소요시간: 177.36초
# # 최종 훈련 데이터 평가 - Loss: 0.5717, Accuracy: 0.6929

# # 모델 학습 완료. 소요시간: 266.41초
# # 최종 훈련 데이터 평가 - Loss: 0.6832, Accuracy: 0.5723


import numpy as np
import pandas as pd
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator

# ✅ 경로 및 설정
np_path = '/workspace/TensorJae/Study25/_save/keras46_men_women/'

# ✅ 데이터 불러오기 + 정규화
start = time.time()
x = np.load(np_path + 'keras46_mw3_x_train.npy')
y = np.load(np_path + 'keras46_mw3_y_train.npy')
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
print(f"데이터 로딩 완료. 소요시간: {round(time.time() - start, 2)}초")
print(x_train.shape, y_train.shape)

# ✅ 데이터 증강 설정
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = datagen.flow(x_train, y_train, batch_size=32)

# ✅ 모델 구성 (성능 개선안)
model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(300,300,3)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# ✅ 손실함수, 옵티마이저, 콜백
loss_fn = BinaryCrossentropy(label_smoothing=0.02)
optimizer = Adam(learning_rate=0.0001) # ✅ 학습률 조정

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['acc'])

es = EarlyStopping(monitor='val_acc', mode='max', patience=15, restore_best_weights=True, verbose=1) # val_acc 모니터링
lr = ReduceLROnPlateau(monitor='val_acc', mode='max', factor=0.5, patience=10, verbose=1) # val_acc 모니터링

mcp_path = '/workspace/TensorJae/Study25/_save/keras46_men_women/'

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only= True,
    filepath = mcp_path + 'keras46_gender.h5'
)
# ✅ 모델 학습 (데이터 증강 적용)
start = time.time()
hist = model.fit(
    train_generator,
    steps_per_epoch=len(x_train) // 32,
    epochs=200,
    validation_data=(x_test, y_test),
    callbacks=[es, lr],
    verbose=1
)
print(f"모델 학습 완료. 소요시간: {round(time.time() - start, 2)}초")

# ✅ 최종 평가
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"최종 테스트 데이터 평가 - Loss: {loss:.4f}, Accuracy: {acc:.4f}")


# 1. x_test에 대한 예측 수행
predictions_proba = model.predict(x_test)  # 확률값 (e.g., 0.98, 0.12, ...)
result_class = np.round(predictions_proba)     # 최종 클래스 (1 또는 0)

# 2. 예측 결과(result_class)와 실제 정답(y_test)을 비교
print("\n===== 상위 20개 샘플 비교 =====")

# 보기 좋게 1차원 배열로 만듭니다.
print("예측 결과 (Predictions):")
print(result_class[:20].flatten().astype(int)) 

print("\n실제 정답 (True Labels):")
print(y_test[:20].astype(int))

# result = np.round(model.predict(x_test))
# print(result[:20])
# print(y_train[:20])

# 모델 학습 완료. 소요시간: 718.72초
# 최종 테스트 데이터 평가 - Loss: 0.4651, Accuracy: 0.8142

# 모델 학습 완료. 소요시간: 1325.05초
# 최종 테스트 데이터 평가 - Loss: 0.5281, Accuracy: 0.8353

# 모델 학습 완료. 소요시간: 3888.31초
# 최종 테스트 데이터 평가 - Loss: 0.5548, Accuracy: 0.8248

# ===== 상위 20개 샘플 비교 =====
# 예측 결과 (Predictions):
# [1 1 0 1 0 0 1 1 0 1 1 1 1 0 0 1 0 1 1 1]

# 실제 정답 (True Labels):
# [1 1 0 0 1 0 1 1 0 1 1 1 1 0 0 1 0 0 1 1]