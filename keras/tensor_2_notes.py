# [] 대괄호가 뜬금없이 있으면 들어갈 수 있는 애들을 쓴거임
n = int     # 아무 숫자
la = int    # label 수
co = 3      # colour
bw = 1      # black & white

from keras.models import Sequential
from keras.models import Model
from keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time

#1. 데이터

# keras 데이터
from keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# 원래는 위와 같이 써야 하지만 설명의 편의를 위해 아래와 같이 작성함
dataset = mnist()
x = dataset.data
y = dataset.target

# print(x.shape)  #(70000, 28, 28)
# print(y.shape)  #(70000, )

# 외부 데이터(아직 배우지 않아서 모름)
path = '___'

# scaling 1 (기존 Scaler 사용법)
# 1-1) 기본 방식(reshape)
x = x.reshape(n, n*n)       # 기존 스케일러는 2차원 데이터 까지만 가능
y = y.reshape(n, n*n)
# 1-2) 멋진 방식(reshape)
x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
y = y.reshape(y.shape[0], y.shape[2] * y.shape[2])
# 2) MinMaxScaler 다른 스케일러도 동일함
mms = MinMaxScaler()
x = mms.fit_transform(x)
y = mms.transform(y)
# 3) 다시 reshape
x = x.reshape(n, n, n, 1) 
y = y.reshape(n, n, n, 1)

# scaling 2 (정규화 1)
x = x/255.                  # '.'은 부동소수점 연산을 위해 쓰는 것
y = y/255.

# scaling 3 (정규화 2)
x = (x-127.5)/127.5
y = (y-127.5)/127.5

# OneHot
# numpy
y = y.reshape(n, n, n, la)
ohe = OneHotEncoder()
y = ohe.fit_transform(y)
# pandas
y = pd.get_dummies(y)
# keras
y = to_categorical(y)

# train_test_split
x_train, y_train, x_test, y_test = train_test_split(
    x, y, train_size=n, shuffle=True, random_state=n,
    stratify=y
)

#2. 모델 구성
# 순차형
model = Sequential()
model.add(Conv2D(filters=n, kernel_size=(n,n), strides=n, input_shape=(n,n,[co, bw])))
model.add(Conv2D(n, (n,n), activation=['tanh', 'relu']))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(n, activation=['tanh', 'relu']))                  #tanh 는 -1~1
model.add(Dense(la, activation=['linear', 'sigmoid', 'softmax']))

# 함수형
input = Input(shape=(n,n,[co, bw]))
conv1 = Conv2D(filters=n, kernel_size=(n,n), strides=n)(input)
conv2 = Conv2D(n, (n,n), activation=['tanh', 'relu'])(conv1)
bat1  = BatchNormalization()(conv2)
drop1 = Dropout(0.2)(bat1)
flat  = Flatten()(drop1)
den1  = Dense(n, activation=['tanh', 'relu'])(flat)
output  = Dense(la, activation=['linear', 'sigmoid', 'softmax'])(den1)
model = Model(inputs=input, outputs=output)

model.summary()
model.save(path + '__.h5')
model = load_model(path + '__.5')

#3. 컴파일, 훈련
model.compile(
    loss=['mse', 'binary_crossentropy', 'categorical_crossentropy'],
    optimizer='adam', metrics=['acc']
)

# EarlyStopping
es = EarlyStopping(
    monitor='val_loss', mode='min',
    verbose=n, patience=n, 
    restore_best_weights=True
)

# ModelCheckpoint
path1 = '____'
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath1 = ''.join([path1, '__', date, '_', filename])
mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto',
    verbose=n, save_best_only=True,
    filepath=filepath1
)

s_time = time.time()
hist = model.fit(
    x_train, y_train, epochs=n, batch_size=n,
    verbose=n, validation_split=n, callbacks=[es, mcp]
)
e_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
results = model.predict(x_test)
y_test_arg = np.argmax(y_test)
results_arg = np.argmax(results)
acc = accuracy_score(y_test_arg, results_arg)

print('Loss :', loss[0])
print('ACC  :', loss[1])
print('ACC2 :', acc)

#5. 보너스 그림
plt.rcParams['__'] = '__'
plt.imshow(x_train[n], 'gray')
plt.show()




