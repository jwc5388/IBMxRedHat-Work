from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import sklearn as sk
import pandas as pd
import numpy as np
from tabnanny import verbose
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score, train_test_split
import warnings
from bayes_opt import BayesianOptimization
import random
import time
from keras.callbacks import EarlyStopping
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor, BaggingClassifier, RandomForestRegressor

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

import os

# 기본 경로 설정
if os.path.exists('/workspace/TensorJae/Study25/'):
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')

np_path = os.path.join(BASE_PATH, '_save/save_npy/')

# 저장된 npy 파일 불러오기
x_train = np.load(np_path + 'keras44cd_x_train.npy')  # (25000, 150, 150, 3)
y_train = np.load(np_path + 'keras44cd_y_train.npy')
x_test = np.load(np_path + 'keras44cd_x_test.npy')    # (12500, 150, 150, 3)
y_test = np.load(np_path + 'keras44cd_y_test.npy')

print(f"x_train shape: {x_train.shape}")
print(f"x_test  shape: {x_test.shape}")

# Train + Test 결합
x = np.concatenate([x_train, x_test], axis=0)  # (37500, 150, 150, 3)
y = np.concatenate([y_train, y_test], axis=0)


print(x.shape)  # (37500, 150, 150, 3)

# exit()

# reshape
x = x.reshape(x.shape[0], 150*150*3)  # (37500, 67500)

# # scaling
# scaler = StandardScaler()
# x = scaler.fit_transform(x)

# PCA 적용
max_components = min(x.shape[0], x.shape[1])  # 37500
print(f"PCA 가능한 최대 n_components 수: {max_components}")

pca = PCA(n_components=max_components)
x_pca = pca.fit_transform(x)

# 누적 설명력 계산
evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)

# 누적 설명력 결과 출력
print('--- 누적 설명력 기준 주성분 수 ---')
print('0.95 이상:', np.argmax(evr_cumsum >= 0.95) + 1)
print('0.99 이상:', np.argmax(evr_cumsum >= 0.99) + 1)
print('0.999 이상:', np.argmax(evr_cumsum >= 0.999) + 1)
if np.any(evr_cumsum >= 1.0):
    print('1.0 이상:', np.argmax(evr_cumsum >= 1.0) + 1)
else:
    print('1.0 이상을 만족하는 주성분 수 없음')