# #불균형한 이진분류



# import numpy as np
# import pandas as pd
# from keras.models import Sequential
# from keras.layers import Dense, BatchNormalization, Dropout
# from keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import roc_auc_score
# import time


# # 1. Load Data
# path = 'Study25/_data/kaggle/santander/'

# train_csv = pd.read_csv(path + 'train.csv')
# test_csv = pd.read_csv(path + 'test.csv')
# submission_csv = pd.read_csv(path + 'sample_submission.csv')

# # 2. Feature/Target 분리
# x = train_csv.drop(['ID_code', 'target'], axis=1)
# y = train_csv['target']
# x_submit = test_csv.drop(['ID_code'], axis=1)

# # 3. Train/Test Split
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=33)

# # 4. 스케일링
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# x_submit = scaler.transform(x_submit)

# # 5. 클래스 불균형 보정
# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
# class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# # 6. 모델 구성
# model = Sequential([
#     Dense(256, input_shape=(200,), activation='relu'),
#     BatchNormalization(),
#     Dropout(0.3),
#     Dense(256, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.3),
#     Dense(256, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.2),
#     Dense(1, activation='sigmoid')
# ])

# # 7. 모델 컴파일
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # 8. 조기 종료 설정
# es = EarlyStopping(
#     monitor='val_loss',
#     mode='auto',
#     patience=20,
#     restore_best_weights=True
# )

# # 9. 모델 훈련
# start = time.time()
# model.fit(x_train, y_train,
#           validation_split=0.2,
#           epochs=10000,
#           batch_size=32,
#           callbacks=[es],
#           class_weight=class_weight_dict,
#           verbose=1)
# end = time.time()

# # 10. 평가
# loss, acc = model.evaluate(x_test, y_test)
# y_pred_prob = model.predict(x_test)
# roc_auc = roc_auc_score(y_test, y_pred_prob)

# print(f"loss: {loss:.4f}")
# print(f"accuracy: {acc:.4f}")
# print(f"roc_auc: {roc_auc:.4f}")
# print(f"걸린 시간: {end - start:.2f}초")

# # 11. 제출 파일 생성
# y_submit = model.predict(x_submit)
# submission_csv['target'] = y_submit
# submission_csv.to_csv(path + 'submission_0608_final.csv')


import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Flatten
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import time

# 0. 시드 고정 (재현성을 위해)
np.random.seed(42)
tf.random.set_seed(42)

# 1. 데이터 로드
path = 'Study25/_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv')
test_csv = pd.read_csv(path + 'test.csv')
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 2. 피처/타겟 분리 및 기본 피처 리스트 생성
y = train_csv['target']
x = train_csv.drop(['ID_code', 'target'], axis=1)
x_submit = test_csv.drop(['ID_code'], axis=1)
original_features = [col for col in x.columns]

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


# exit()
# y_test = y_test.reshape(-1,1)
x_train = x_train.to_numpy().reshape(-1, 20,10, 1)
x_test = x_test.to_numpy().reshape(-1, 20,10, 1)
y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)


# 4. Build Model
model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=(20,10, 1), activation='relu', padding='same'))
model.add(BatchNormalization())
# model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
# model.add(MaxPooling2D())
model.add(Dropout(0.3))

# model.add(Conv2D(128, (3, 3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D())
# model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))

model.summary()

# exit()

model.compile(loss = 'mse', optimizer= 'adam', metrics= ['mae'])


es = EarlyStopping(
    monitor='val_loss',       # 기준: 검증 손실
    patience=10,              # 10 epoch 개선 없으면 멈춤
    mode='min',               # 손실이므로 'min'
    verbose=1,
    restore_best_weights=True
)


start = time.time()

# 2. model.fit()에 callbacks 인자로 추가
hist = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[es]  # 👈 여기에 추가!
)



end = time.time()

loss, mae = model.evaluate(x_test, y_test)
print('loss:', loss)
print('mae:', mae)
print('걸린시간:', end - start)