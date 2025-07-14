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

#1 data

# 1. 데이터 로드
path = basepath + '_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv')
test_csv = pd.read_csv(path + 'test.csv')
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 2. 피처/타겟 분리 및 기본 피처 리스트 생성
y = train_csv['target']
x = train_csv.drop(['ID_code', 'target'], axis=1)
x_submit = test_csv.drop(['ID_code'], axis=1)
original_features = [col for col in x.columns]


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42, stratify=y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x = np.concatenate([x_train, x_test], axis=0)

print(x.shape) #(200000, 200)

# exit()


pca = PCA(n_components=200)
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)

print(evr_cumsum)

print('0.95 이상 :', np.argmax(evr_cumsum>= 0.95)+1)
print('0.99 이상 :', np.argmax(evr_cumsum>= 0.99)+1)
print('0.999 이상 :', np.argmax(evr_cumsum>= 0.999)+1)
if np.any(evr_cumsum >= 1.0):
    print('1.0 이상 :', np.argmax(evr_cumsum >= 1.0) + 1)
else:
    print('1.0 이상을 만족하는 주성분 수 없음')
    

# 0.95 이상 : 190
# 0.99 이상 : 198
# 0.999 이상 : 200
# 1.0 이상 : 200