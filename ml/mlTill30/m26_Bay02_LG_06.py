from tabnanny import verbose
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor, XGBClassifier
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

#1 data

# === Load Data ===
path = basepath + '_data/dacon/cancer/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# === Separate Features and Target Label ===
x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']


# === One-Hot Encode Categorical Columns ===
categorical_cols = x.select_dtypes(include='object').columns
x = pd.get_dummies(x, columns=categorical_cols)
test_csv = pd.get_dummies(test_csv, columns=categorical_cols)
x, test_csv = x.align(test_csv, join='left', axis=1, fill_value=0)

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

    model = LGBMClassifier(
        **params, n_jobs=-5,early_stopping_rounds=1
    )

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
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



# ⏱ Optimization Time: 17.92 seconds
# 🚀 Best Parameters Found:
# {'target': 0.88022028453419, 'params': {'n_estimators': 203.51199264000678, 'learning_rate': 0.06658970615104422, 'max_depth': 5.181977532625877, 'min_child_weight': 26.48333303771273, 'subsample': 0.7733551396716398, 'gamma': 0.9242722776276352, 'colsample_bytree': 0.9847923138822793, 'colsample_bylevel': 0.8875664116805573, 'max_bin': 470.29398030801684, 'reg_lambda': 89.48273504276489, 'reg_alpha': 5.983020788322741}}


# ⏱ Optimization Time: 18.95 seconds
# 🚀 Best Parameters Found:
# {'target': 0.8804497475906379, 'params': {'n_estimators': 279.4587015082709, 'learning_rate': 0.08541645028714857, 'max_depth': 3.685459445633432, 'min_child_weight': 6.126103195653394, 'subsample': 0.622418145104171, 'gamma': 2.6823251278497438, 'colsample_bytree': 0.8098964623012505, 'colsample_bylevel': 0.529241126053339, 'max_bin': 19.424835589162214, 'reg_lambda': 75.73503200569536, 'reg_alpha': 3.4805066618466847}}