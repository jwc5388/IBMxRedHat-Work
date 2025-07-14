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

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

#1 data

#1 datafrom sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# x, y 불러오기
x = np.load(basepath + '_save/men_women/keras_mw_x_train.npy')
y = np.load(basepath + '_save/men_women/keras_mw_y_train.npy')

print(x.shape)  #(3309, 150, 150, 3)
# exit()
# reshape
x = x.reshape(x.shape[0], 150*150*3)

# scaling
scaler = StandardScaler()
x = scaler.fit_transform(x)

max_components = min(x.shape[0], x.shape[1])
print(f"PCA 가능한 최대 n_components 수: {max_components}")

# exit()
# PCA 적용
pca = PCA(n_components=3300)
x_pca = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)

# 누적 설명력 출력
print('0.95 이상:', np.argmax(evr_cumsum >= 0.95) + 1)
print('0.99 이상:', np.argmax(evr_cumsum >= 0.99) + 1)
print('0.999 이상:', np.argmax(evr_cumsum >= 0.999) + 1)
if np.any(evr_cumsum >= 1.0):
    print('1.0 이상:', np.argmax(evr_cumsum >= 1.0) + 1)
else:
    print('1.0 이상을 만족하는 주성분 수 없음')

# PCA 가능한 최대 n_components 수: 3309
# 0.95 이상: 803
# 0.99 이상: 2074
# 0.999 이상: 3022
# 1.0 이상을 만족하는 주성분 수 없음