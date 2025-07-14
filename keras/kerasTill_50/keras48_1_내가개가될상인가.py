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

# ✅ 경로 설정
np_path = '/workspace/TensorJae/Study25/_save/save_npy/'
path_test = '/workspace/TensorJae/Study25/_data/kaggle/cat_dog/test2/'
# sample_path = '/workspace/TensorJae/Study25/_data/kaggle/cat_dog/sample_submission.csv'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# save_path = f'/workspace/TensorJae/Study25/_save/submission_{timestamp}.csv'

# ✅ 데이터 불러오기 + 정규화
start = time.time()
x = np.load(np_path + 'save_npykeras44_x_train_new.npy') / 255.0
y = np.load(np_path + 'save_npykeras44_y_train_new.npy')
test = np.load(np_path + 'save_npykeras44_x_test.npy') / 255.0
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=42)
print(f"데이터 로딩 완료. 소요시간: {round(time.time() - start, 2)}초")
print(x_train.shape, y_train.shape)

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


# ✅ 모델 구성
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(150,150,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))  # 🔹 약하게

model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))  # 🔹 제일 깊은 층만 살짝 강하게

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# ✅ 손실함수, 옵티마이저, 콜백
loss_fn = BinaryCrossentropy(label_smoothing=0.02)
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1)


mcp_path = '/workspace/TensorJae/Study25/_save/keras48_doghuman/'
mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,
    filepath = mcp_path + 'keras46_dh.h5'
)



# ✅ 모델 학습
start = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[es, lr, mcp],
    verbose=1
)
print(f"모델 학습 완료. 소요시간: {round(time.time() - start, 2)}초")

# ✅ 평가
loss, acc = model.evaluate(x_train, y_train, verbose=0)
print(f"최종 훈련 데이터 평가 - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# ✅ 예측 및 제출 파일 생성
pred = model.predict(test, verbose=1)
pred_prob = pred.reshape(-1)

# submission = pd.read_csv(sample_path)
# submission['label'] = pred_prob
# submission.to_csv(save_path, index=False)
# print(f"✅ 제출 파일 저장 완료: {save_path}")
# ✅ me 이미지 불러오기
me_img = np.load('/workspace/TensorJae/Study25/_data/image/me/keras47_me.npy')

# ✅ 사이즈가 (100,100,3)이므로 모델에 맞게 (150,150,3)로 리사이즈
me_img_resized = tf.image.resize(me_img, [150, 150])
me_img_resized = me_img_resized / 255.0  # 정규화

# ✅ 예측
me_pred = model.predict(me_img_resized)
print(f"🧠 me.jpeg 예측 확률: {me_pred[0][0]:.4f}")
print("✅ 예측 결과:", "🐱 고양이 (cat)" if me_pred[0][0] >= 0.5 else "🐶 개 (dog)")

# ✅ 예측 결과: 🐶 개 (dog)