
from sklearn.datasets import load_diabetes, load_breast_cancer, load_wine, fetch_covtype, load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
#1data

seed = 42

random.seed(seed)
np.random.seed(seed)

import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

# 1. Load digits dataset
dataset = load_digits()
x = dataset.data  # shape: (1797, 64)
y = dataset.target  # shape: (1797,)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42, stratify=y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



es = xgb.callback.EarlyStopping(
    rounds = 50, 
    # metric_name = 'mlogloss',
    data_name = 'validation_0',
    # save_best = True,
    
)

model = XGBClassifier(
    n_estimators = 500,
    random_state = seed,
    gamma = 0,
    min_child_weight = 0,
    reg_alpha = 0,
    reg_lambda = 1,
    # eval_metric = 'mlogloss', # 다중분류: mlogloss, merror, 이진분류: logloss, error
    #                         # 2.1.1 버전 이후로 fit에서 모델로 위치이동 
    callbacks = [es]
    
    )


model.fit(x_train, y_train, eval_set =[(x_test,y_test)], verbose=1)




print('acc:', model.score(x_test, y_test))   # acc2: 0.9333333333333333
print(model.feature_importances_)# [0.01230742 0.02487084 0.5794107  0.38341108]


thresholds = np.sort(model.feature_importances_) #오름차순
print(thresholds) #[0.01230742 0.02487084 0.38341108 0.5794107 ]

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit = False)
    # threshold 가 i값 이상인것을 모두 훈련시킨다. 
    # prefit = False : 모델이 아직 학습 되지 않았을때, model.fit 호출해서 훈련한다. (기본)
    # prefit = True : 이미 학습 된 모델을 전달 할 때, model.fit
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(select_x_train.shape) 

    select_model = XGBClassifier(
        n_estimators = 500,
        random_state = seed,
        gamma = 0,
        min_child_weight = 0,
        reg_alpha = 0,
        reg_lambda = 1,
        # eval_metric = 'mlogloss', # 다중분류: mlogloss, merror, 이진분류: logloss, error
        #                         # 2.1.1 버전 이후로 fit에서 모델로 위치이동 
        callbacks = [es]
         
        )
    
    select_model.fit(select_x_train, y_train, eval_set =[(select_x_test,y_test)], verbose=0)
    
    select_y_pred = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_pred)
    print('Trech=%.3f, n=%d, acc: %.4f%%' %(i, select_x_train.shape[1], score*100))



# Trech=0.000, n=64, acc: 95.0000%
# Trech=0.000, n=64, acc: 85.0000%
# Trech=0.000, n=64, acc: 85.0000%
# Trech=0.000, n=64, acc: 85.0000%
# Trech=0.000, n=64, acc: 85.0000%
# Trech=0.000, n=64, acc: 85.0000%
# Trech=0.000, n=58, acc: 85.0000%
# Trech=0.000, n=57, acc: 85.0000%
# Trech=0.000, n=56, acc: 85.0000%
# Trech=0.000, n=55, acc: 85.0000%
# Trech=0.001, n=54, acc: 85.0000%
# Trech=0.001, n=53, acc: 85.0000%
# Trech=0.001, n=52, acc: 85.0000%
# Trech=0.002, n=51, acc: 85.0000%
# Trech=0.002, n=50, acc: 85.0000%
# Trech=0.003, n=49, acc: 85.0000%
# Trech=0.004, n=48, acc: 84.4444%
# Trech=0.004, n=47, acc: 84.4444%
# Trech=0.005, n=46, acc: 84.4444%
# Trech=0.005, n=45, acc: 84.4444%
# Trech=0.005, n=44, acc: 84.4444%
# Trech=0.005, n=43, acc: 84.1667%
# Trech=0.006, n=42, acc: 84.4444%
# Trech=0.006, n=41, acc: 85.0000%
# Trech=0.006, n=40, acc: 84.1667%
# Trech=0.006, n=39, acc: 84.1667%
# Trech=0.006, n=38, acc: 84.1667%
# Trech=0.007, n=37, acc: 85.0000%
# Trech=0.007, n=36, acc: 85.5556%
# Trech=0.007, n=35, acc: 85.5556%
# Trech=0.007, n=34, acc: 87.2222%
# Trech=0.007, n=33, acc: 87.2222%
# Trech=0.008, n=32, acc: 87.2222%
# Trech=0.008, n=31, acc: 86.9444%
# Trech=0.009, n=30, acc: 86.6667%
# Trech=0.009, n=29, acc: 87.2222%
# Trech=0.009, n=28, acc: 88.3333%
# Trech=0.010, n=27, acc: 89.4444%
# Trech=0.011, n=26, acc: 89.1667%
# Trech=0.013, n=25, acc: 89.4444%
# Trech=0.014, n=24, acc: 88.8889%
# Trech=0.015, n=23, acc: 88.0556%
# Trech=0.016, n=22, acc: 87.5000%
# Trech=0.017, n=21, acc: 86.9444%
# Trech=0.018, n=20, acc: 87.5000%
# Trech=0.018, n=19, acc: 87.2222%
# Trech=0.019, n=18, acc: 86.9444%
# Trech=0.020, n=17, acc: 87.5000%
# Trech=0.023, n=16, acc: 86.3889%
# Trech=0.025, n=15, acc: 87.2222%
# Trech=0.026, n=14, acc: 85.2778%
# Trech=0.027, n=13, acc: 86.3889%
# Trech=0.030, n=12, acc: 86.1111%
# Trech=0.034, n=11, acc: 86.1111%
# Trech=0.038, n=10, acc: 82.5000%
# Trech=0.038, n=9, acc: 82.2222%
# Trech=0.039, n=8, acc: 79.7222%
# Trech=0.045, n=7, acc: 77.7778%
# Trech=0.046, n=6, acc: 72.7778%
# Trech=0.054, n=5, acc: 72.5000%
# Trech=0.059, n=4, acc: 57.7778%
# Trech=0.064, n=3, acc: 46.3889%
# Trech=0.066, n=2, acc: 37.5000%
# Trech=0.070, n=1, acc: 24.7222%