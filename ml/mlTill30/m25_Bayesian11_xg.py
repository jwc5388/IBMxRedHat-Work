from tabnanny import verbose
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from sklearn.datasets import fetch_california_housing, load_digits
from sklearn.model_selection import cross_val_score, train_test_split
import warnings
from bayes_opt import BayesianOptimization
import random
import time
from keras.callbacks import EarlyStopping
import pandas as pd


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

# 1. Load digits dataset
digits = load_digits()
x = digits.data  # shape: (1797, 64)
y = digits.target  # shape: (1797,)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42, stratify=y)


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

from xgboost.callback import EarlyStopping

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

    model = XGBClassifier(**params)

    from xgboost.callback import EarlyStopping

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        eval_metric='mlogloss',   # 또는 'mae', 'logloss', 'error' 등
        callbacks=[
            EarlyStopping(rounds=10, save_best=True)
        ],
        verbose=False
    )

    y_pred = model.predict(x_test)
    result = accuracy_score(y_test, y_pred)
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