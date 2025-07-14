# 재료
n = int
np.set_printoptions(threshold=np.inf) # 큰 데이터 생략없이 다보기

#region(#0. 임포트)

from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
import time
import random as rd
import matplotlib.pyplot as plt
import datetime
# 그 외
from sklearn.datasets import fetch_california_housing # 선형 회귀
from sklearn.datasets import load_breast_cancer       # 이진 분류
from sklearn.datasets import load_iris                # 다중 분류
from sklearn.datasets import fetch_covtype
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_boston              # 윤리적 문제로 deprecated (boston 가상환경에서만 사용할 수 있음)
from keras.datasets import mnist

#endregion

#region(#1. 데이터)

#region(데이터 불러오기)
# sklearn 자료
datasets = load_iris()
# 관련 명령어
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)
x_iris = datasets.data
y_iris = datasets.target
# 관련 명령어
print(x_iris.shape)
print(y_iris.shape)

# 외부 자료
path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sampleSubmission.csv')
# # 관련 명령어
print(train_csv)
print(train_csv.shape)
print(train_csv.columns)
print(type(train_csv))
print(train_csv.info())
print(train_csv.describe())
#endregion

#region(결측치 처리)
print(train_csv.isna().sum())
train_csv['___'] = train_csv['___'].replace(0, np.nan)
train_csv['___'] = train_csv['___'].dropna()
train_csv['___'] = train_csv['___'].fillna(train_csv.mean())
train_csv['___'] = train_csv['___'].fillna(train_csv.median())
train_csv['___'] = train_csv['___'].fillna(train_csv.max())
train_csv['___'] = train_csv['___'].fillna(train_csv.min())
#endregion

#region(Encoder)
le = LabelEncoder()
train_csv[['___']] = le.fit_transform(train_csv[['___']])                   # 문자열을 숫자로 변환(열 1개만 처리가능)

oe = OrdinalEncoder()
train_csv[['___', '___']] = oe.fit_transform(train_csv[['___', '___']])     # 문자열을 숫자로 변환(여러 개 열 처리 가능)

# 1) numpy
ohe = OneHotEncoder(sparse=False)
y_iris = y_iris.reshape(-1,1)
train_csv[['___', '___']] = ohe.fit_transform(train_csv[['___', '___']])    # 숫자 행렬 데이터로 변환(지금은 y에서만 하지만 나중에는 x에서도 함)
                                                                            # output_layer의 node의 개수는 OneHotEncoder()한 label의 갯수
# 2) pandas
y_iris = pd.get_dummies(y_iris)

# 3) keras
from keras.utils import to_categorical
y_iris = to_categorical(y_iris)

#endregion

#region(변수에 담기)
x = train_csv.drop(['___', '___'], axis=1)
y = train_csv['___', '___']
#endregion

#region(train_test_split)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, test_size=0.25,
    shuffle=True, random_state=rd.randint(1,1000000),
    stratify=y
)
#endregion

#region(reshape)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
#endregion

#region(Scaler)
# StandartScaler
ss = StandardScaler()
ss.fit(x_train[['___']])
x_train[['___']] = ss.transform(x_train[['___']])
x_test[['___']] = ss.transform(x_test[['___']])
test_csv[['___']] = ss.transform(test_csv[['___']])

# MinMaxScaler
mms = MinMaxScaler()
mms.fit(x_train[['___']])
x_train[['___']] = mms.transform(x_train[['___']])
x_test[['___']] = mms.transform(x_test[['___']])
test_csv[['___']] = mms.transform(test_csv[['___']])

# MaxAbsScaler
mas = MaxAbsScaler()
mas.fit(x_train[['___']])
x_train[['___']] = mas.transform(x_train[['___']])
x_test[['___']] = mas.transform(x_test[['___']])
test_csv[['___']] = mas.transform(test_csv[['___']])

# RobustScaler
rs = RobustScaler()
rs.fit(x_train[['___']])
x_train[['___']] = rs.transform(x_train[['___']])
x_test[['___']] = rs.transform(x_test[['___']])
test_csv[['___']] = rs.transform(test_csv[['___']])

#endregion
#endregion

#region(#2. 모델구성)
# 순차형 모델
model = Sequential()
model.add(Dense(n, input_dim=n, activation='relu'))     # input_dim은 매트릭스 데이터용
model.add(Dense(n, input_shape=(n,)))                  # 행무시 열우선 텐서 데이터는 shape ex) (10000, 10, 3)이면 input_shape=(10,3)
model.add(BatchNormalization())                         # BatchNorm
model.add(Dropout(0.3))                                 # Dropout(0.3~0.5가 적당)
model.add(Dense(n, activation='linear'))                # 선형 회귀
model.add(Dense(1, activation='sigmoid'))               # 이진 분류
model.add(Dense(n, activation='softmax'))               # 다중 분류 n은 y 라벨의 개수

# 함수형 모델
input = Input(shape=(n,))                               # 행무시 열우선으로 (열,) 형태
dense1= Dense(n, name='__')
drop1 = Dropout(0.3)
dense2= Dense(n, name='__', activation='relu')
output= Dense(n)
model = Model(inputs= input, ouputs=output)

# CNN 모델
# 커널 사이즈로 데이터의 사이즈를 점점 축소시킴
# 너무 많이 자르면 너무 소실되면서 loss가 커짐
# 너무 조금 자르면 특성값이 크지 않아서 loss가 커짐
# 그러니까 적당히 자르는 것이 하이퍼 파라미터 튜닝임
# !!! CNN의 커널 사이즈는 weight다. !!!
model = Sequential()
model.add(Conv2D(filters=n, kernel_size=(n,n), 
                 strides=n, input_shape=(n,n,1)))       # 흑백 (n,n,1)
model.add(Conv2D(filters=n, kernel_size=(n,n),          
                 strides=n, input_shape=(n,n,3)))       # 컬러
model.add(Conv2D(n, (n,n)))
model.add(Flatten())                                    # 2차원으로 바꿔줌
model.add(Dense(n, activation='relu'))
model.add(Dense(n, activation=['linear','sigmoid','softmax']))

model.summary()                                         # 연산 횟수 계산
model.save(path + '___.h5')                             # 모델과 초기 랜덤 가중치를 저장
model = load_model(path + '___.h5')                     # 모델과 초기 랜덤 가중치를 로드
model.save_weight(path + '___.h5')                      # 가중치 값 '만' 저장
model.load_weight(path + '___.h5')                      # 훈련한 가중치를 불러와야 함
model = load_model(path + '___.hdf5')                   # ModelCheckpoint를 로드(모델과 컴파일, 훈련까지 모두 저장)

#endregion

#region(#3. 컴파일 훈련)
# 컴파일
model.compile(
    loss='mse', optimizer='adam', metrics=['acc']
)   # 선형 회귀
model.compile(
    loss='binary_crossentropy', optimizer='adam', metrics=['acc']
)   # 이진 분류
model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['acc']
)   # 다중 분류

# EarlyStopping
es = EarlyStopping(
    monitor='val_loss', mode='min',                     # 'max' , 'auto'
    patience=n, restore_best_weights=True               # default: False
)
# ModelCheckpoint
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filename = '{epoch:04d}-{val_loss:.4f}.h5'
filepath = ''.join([path, 'k27_', date, '_', filename])
mcp = ModelCheckpoint(                                  # 항상 사용
    monitor='val_loss', mode='auto',
    save_best_only=True, 
    filepath=filepath,
)
# ReduceLROnPlateau
rl = ReduceLROnPlateau(
    monitor='val_loss', mode='auto',
    patience=10, verbose=2,
    factor=0.5
)

# 훈련
start_time = time.time()
hist = model.fit(
    x_train, y_train, epochs=n, batch_size=n,
    verbose=n, validation_split=0.2,
    callbacks=[es, mcp, rl]                             # es = EarlyStopping, mcp = ModelCheckpoint, rl = ReduceLROnPlateau
)
end_time = time.time()

# 관련 명령어
print(hist.history)

model.save(path + '___.h5')                             # 모델+컴파일훈련 까지 저장
model = load_model(path + '___.h5')                     # 모델+컴파일훈련 까지 로드
model.save_weight(path + '___.h5')                      # 가중치 값 '만' 저장
#endregion

#region(#4. 평가, 예측)
# 평가
loss = model.evaluate(x_test, y_test)
# 예측
results = model.predict(x_test)

results = np.round(results)                             # 반올림 해야 할 때
# 평가 지표
r2 = r2_score(y_test, results)
f1 = f1_score(y_test, results, average='macro')         # f1은 이진에서 f1 & average 는 다중에서
acc = accuracy_score(y_test, results)
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse_1 = RMSE(y_test, results)
rmse_2 = np.sqrt(loss)

# 출력
print('Loss :', loss)
print('RMSE :', rmse_1)                                 # 선형 회귀
print('RMSE :', rmse_2)                                 # 선형 회귀
print('R2   :', r2)                                     # 선형 회귀
print('Acc  :', acc)                                    # 이진 분류 / 다중 분류
print('Acc  :', loss[1])
print('F1   :', f1)                                     # 이진 분류 / 다중 분류
print('time :', np.round(end_time - start_time, 1), 'sec')  # 걸린 시간 / np.round() : 반올림

#endregion

#region(#bonus. 그림 그리기)
plt.rcParams['__.____'] = '___'
plt.figure(figsize=(n,n))
plt.plot(hist.history['___'], c='___', label='___')
plt.plot(hist.history['___'], c='___', label='___')
plt.title('___')
plt.xlabel('___')
plt.ylabel('___')
plt.legend(loc='___')
plt.grid()
plt.show()

#endregion

#region(python 기초)
aaa = [1,2,3,4,5]
for i in aaa:       # for 반복문
    print(i)

#endregion