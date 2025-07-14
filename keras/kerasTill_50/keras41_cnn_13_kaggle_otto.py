# import time
# import pandas as pd
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, BatchNormalization, Dropout
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from keras.utils import to_categorical
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# # 1. Load Data
# path = 'Study25/_data/kaggle/otto/'
# train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# submission_csv = pd.read_csv(path + 'sampleSubmission.csv')


# # print(train_csv) #  [61878 rows x 94 columns]
# # print(train_csv.info())  
# # print(test_csv) #[144368 rows x 93 columns]
# # print(test_csv.info())

# # 2. Feature & Target 분리
# x = train_csv.drop(['target'], axis=1)
# y = train_csv['target']

# print(y)
# print(np.unique(y, return_counts=True))
# # (array(['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6',
# #        'Class_7', 'Class_8', 'Class_9'], dtype=object), array([ 1929, 16122,  8004,  2691,  2739, 14135,  2839,  8464,  4955]))
# # exit()

# # 3. 라벨 인코딩 + 원핫 인코딩
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# y_ohe = pd.get_dummies(y_encoded)
# # y_ohe = to_categorical(y_encoded)

# # 4. Train/Test split
# x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, train_size=0.8, random_state=42)

# # 5. 스케일링
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# x_submit = scaler.transform(test_csv)

# # 6. 모델 정의
# model = Sequential([
#     Dense(512, input_dim=93, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.5),

#     Dense(256, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.4),

#     Dense(128, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.3),

#     Dense(9, activation='softmax')
# ])

# # 7. 컴파일
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # 8. 콜백 설정
# es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
# rl = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

# start = time.time()
# # 9. 훈련
# model.fit(
#     x_train, y_train,
#     validation_split=0.2,
#     epochs=200,
#     batch_size=64,
#     callbacks=[es, rl],
#     verbose=1
# )
# end = time.time()

# # 10. 평가
# loss, acc = model.evaluate(x_test, y_test)
# print(f'loss: {loss:.4f}, accuracy: {acc:.4f}')
# print('걸린 시간:', end-start)


# # 11. 예측 및 제출파일 저장
# preds = model.predict(x_submit)
# submission_df = pd.DataFrame(preds, columns=submission_csv.columns[1:])
# submission_df.insert(0, 'id', submission_csv['id'])
# submission_df.to_csv(path + 'otto_submission.csv', index=False)
# print("✅ 제출 파일 저장 완료: otto_submission.csv")


import time
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# 1. Load Data
path = 'Study25/_data/kaggle/otto/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

# 2. Feature & Target 분리
x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

print(x.shape)
print(y.shape)
# exit()

# 4. Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 5. Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 6. Reshape for CNN input
x_train = x_train.reshape(-1, 31, 3, 1)
x_test = x_test.reshape(-1, 31, 3, 1)

# 7. Build Model
model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(31,3,1)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))  # ✅ 다중분류이므로 softmax + 9개 클래스

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


# accuracy: 0.8068 - loss: 0.5071
# loss: 0.5122, accuracy: 0.8063, 걸린시간: 332.19초