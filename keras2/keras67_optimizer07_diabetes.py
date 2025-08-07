import sklearn as sk
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from keras.optimizers import Adam, Adagrad, SGD, RMSprop
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

#1 data
path = basepath +  '_data/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col= 0)

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())
print(x.shape)
print(y.shape)

# exit()

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=SEED, stratify=y)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# 1. 데이터 준비
# dataset = fetch_california_housing()
# x = dataset.data
# y = dataset.target
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333)

# 2. 탐색할 옵티마이저 및 러닝레이트
optimizers = [Adam, Adagrad, SGD, RMSprop]
lr_list = [0.1, 0.01, 0.05, 0.001, 0.0005, 0.0001]

# 3. 결과 저장용 리스트
results = []

# 4. 탐색
for opt in optimizers:
    for lr in lr_list:
        model = Sequential([
            Dense(128, input_dim=8, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1)
        ])

        optimizer_instance = opt(learning_rate=lr)
        model.compile(loss='mse', optimizer=optimizer_instance)

        start = time.time()
        history = model.fit(
            x_train, y_train,
            epochs=10,
            batch_size=32,
            verbose=0,
            validation_split=0.2
        )
        end = time.time()

        final_val_loss = history.history['val_loss'][-1]
        results.append({
            'optimizer': opt.__name__,
            'lr': lr,
            'val_loss': final_val_loss,
            'time': end - start
        })

        print(f"[{opt.__name__} | lr={lr}] → val_loss: {final_val_loss:.6f} / time: {end - start:.2f} sec")

# 5. 최적 조합 출력
best = sorted(results, key=lambda x: x['val_loss'])[0]
print("\n✅ 가장 좋은 결과:")
print(f"Optimizer: {best['optimizer']}, Learning Rate: {best['lr']}")
print(f"Validation Loss: {best['val_loss']:.6f}, Time: {best['time']:.2f} sec")

# ✅ 가장 좋은 결과:
# Optimizer: Adam, Learning Rate: 0.001
# Validation Loss: 0.376366, Time: 0.65 sec