from tabnanny import verbose
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score, train_test_split
import warnings
from bayes_opt import BayesianOptimization
import random
import time
from keras.callbacks import EarlyStopping
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor


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


path = basepath + '_data/dacon/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
# print(train_csv) #[1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
# print(test_csv) #[715 rows x 9 columns]

submission_csv = pd.read_csv(path+ 'submission.csv', index_col=0)
# print(submission_csv) # [715 rows x 1 columns]

train_csv = train_csv.dropna()


# test_csv = test_csv.dropna()
test_csv = test_csv.fillna(train_csv.mean())

x = train_csv.drop(['count'], axis=1)  #count라는 axis=1 열 삭제, 행은 axis =0
print(x) #[1459 rows x 9 columns]

y = train_csv['count'] 
print(y) #(1459,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 베이지안 탐색 범위 설정
bayesian_params = {
     "n_estimators" : (100,500),
    'learning_rate': (0.001, 0.1),
    'max_depth': (3, 10),
    'min_child_weight': (1, 50),
    'subsample': (0.5, 1),
    'gamma': (0,5),
    'colsample_bytree': (0.5, 1),
    "colsample_bylevel" : (0.5,1),
    
    'max_bin': (9, 500),
    'reg_lambda': (0, 100),         #default 1// L2 정규화// 릿지
    'reg_alpha': (0.01, 10),        #default 0// L1 정규화// 라쏘
}


def xgb_cv(n_estimators, learning_rate, max_depth, min_child_weight, gamma, 
           subsample, colsample_bytree, colsample_bylevel, max_bin, reg_lambda, reg_alpha):
    params = {
        "n_estimators": int(n_estimators),
        "learning_rate": learning_rate,
        "max_depth": int(round(max_depth)),
        "min_child_weight": min_child_weight,
        "gamma": gamma,
        "subsample": min(max(subsample, 0), 1),
        "colsample_bytree": min(max(colsample_bytree, 0), 1),
        "colsample_bylevel": min(max(colsample_bylevel, 0), 1),
        "max_bin": int(round(max_bin)),
        "reg_lambda": max(reg_lambda, 0),
        "reg_alpha": reg_alpha,
        "random_state": 42,
        "tree_method": "hist",  # or 'gpu_hist' if using GPU
    }

    model = LGBMRegressor(
        **params, n_jobs=-5,early_stopping_rounds=1
    )

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        verbose=False
    )

    y_pred = model.predict(x_test)
    result = r2_score(y_test, y_pred)
    return result
# 베이지안 옵티마이저 실행
optimizer = BayesianOptimization(
    f=xgb_cv,
    pbounds=bayesian_params,
    random_state=42,
    # verbose=1,
)

start = time.time()
optimizer.maximize(init_points=5, n_iter=30)
end = time.time()

print('⏱ Optimization Time:', round(end - start, 2), 'seconds')
print('🚀 Best Parameters Found:')
print(optimizer.max)


# ⏱ Optimization Time: 24.5 seconds
# 🚀 Best Parameters Found:
# {'target': 0.8042358773425258, 'params': {'n_estimators': 276.614821771048, 'learning_rate': 0.1, 'max_depth': 10.0, 'min_child_weight': 1.0, 'subsample': 1.0, 'gamma': 0.0, 'colsample_bytree': 1.0, 'colsample_bylev


# ⏱ Optimization Time: 22.45 seconds
# 🚀 Best Parameters Found:
# {'target': 0.7953992805255825, 'params': {'n_estimators': 482.77237438976886, 'learning_rate': 0.1, 'max_depth': 9.491537508209074, 'min_child_weight': 16.827474027591276, 'subsample': 0.5, 'gamma': 5.0, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.9322337122008553, 'max_bin': 264.82207282773453, 'reg_lambda': 0.0, 'reg_alpha': 10.0}}