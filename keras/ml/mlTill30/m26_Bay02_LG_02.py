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


warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

# 데이터 로드 및 전처리
x, y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

scaler = MinMaxScaler()
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

# |   iter    |  target   | n_esti... | learni... | max_depth | min_ch... | subsample |   gamma   | colsam... | colsam... |  max_bin  | reg_la... | reg_alpha |
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# | 24        | 0.8326435 | 265.85648 | 0.1       | 5.3601965 | 9.5977450 | 1.0       | 0.0       | 1.0       | 0.8018240 | 400.33050 | 60.855757 | 10.0      |
# | 27        | 0.8398038 | 271.52497 | 0.1       | 10.0      | 49.798952 | 0.9992568 | 0.0       | 0.5       | 1.0       | 385.39257 | 91.764114 | 10.0      |
# | 30        | 0.8472374 | 291.58749 | 0.1       | 10.0      | 1.0       | 1.0       | 0.0       | 0.5       | 1.0       | 402.56017 | 14.203701 | 0.01      |
# | 37        | 0.8490052 | 321.62065 | 0.1       | 10.0      | 34.195884 | 1.0       | 0.0       | 1.0       | 0.5418226 | 372.15500 | 72.518143 | 10.0      |
# | 46        | 0.8502464 | 367.65366 | 0.1       | 10.0      | 11.322973 | 1.0       | 0.0       | 1.0       | 1.0       | 395.30472 | 18.186788 | 10.0      |
# | 59        | 0.8523646 | 352.26332 | 0.1       | 9.0370039 | 11.249666 | 1.0       | 0.0       | 1.0       | 0.9109928 | 369.93508 | 0.0       | 10.0      |
# | 108       | 0.8535246 | 419.24668 | 0.1       | 10.0      | 4.8702766 | 1.0       | 0.0       | 0.5       | 1.0       | 358.29244 | 0.0       | 10.0      |
# | 150       | 0.8552771  494.47022 || 0.1       | 10.0      | 37.830349 | 1.0       | 0.0       | 1.0       | 0.5       | 301.49274 | 27.983786 | 0.01      |
# =============================================================================================================================================================
# ⏱ Optimization Time: 430.06 seconds
# 🚀 Best Parameters Found:
# {'target': 0.8552771144574414, 'params': {'n_estimators': 494.47022000418707, 'learning_rate': 0.1, 'max_depth': 10.0, 'min_child_weight': 37.83034934293349, 'subsample': 1.0, 'gamma': 0.0, 'colsample_bytree': 1.0, 'colsample_bylevel': 0.5, 'max_bin': 301.49274039675765, 'reg_lambda': 27.983786391559438, 'reg_alpha': 0.01}}